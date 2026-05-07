// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpuevent // import "go.opentelemetry.io/obi/pkg/internal/ebpf/gpuevent"

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync"

	"github.com/cilium/ebpf"
	"github.com/gavv/monotime"

	"go.opentelemetry.io/obi/pkg/appolly/app"
	"go.opentelemetry.io/obi/pkg/appolly/app/request"
	"go.opentelemetry.io/obi/pkg/appolly/app/svc"
	"go.opentelemetry.io/obi/pkg/appolly/discover/exec"
	ebpfcommon "go.opentelemetry.io/obi/pkg/ebpf/common"
	"go.opentelemetry.io/obi/pkg/export/imetrics"
	"go.opentelemetry.io/obi/pkg/internal/ebpf/ringbuf"
	"go.opentelemetry.io/obi/pkg/internal/goexec"
	"go.opentelemetry.io/obi/pkg/obi"
	"go.opentelemetry.io/obi/pkg/pipe/msg"
)

//go:generate $BPF2GO -cc $BPF_CLANG -cflags $BPF_CFLAGS -type cuda_kernel_launch_t -type cuda_graph_launch_t -type cuda_malloc_t -type cuda_memcpy_t -type cuda_sync_t -type cuda_free_t -type cuda_memset_t -type cuda_peer_copy_t -type cuda_kernel_launch_done_t -type cuda_error_t -type hami_oom_t -type hami_throttle_t -target amd64,arm64 Bpf ../../../../bpf/gpuevent/gpuevent.c -- -I../../../../bpf

const (
	EventTypeKernelLaunch     = 1  // EVENT_CUDA_KERNEL_LAUNCH
	EventTypeMalloc           = 2  // EVENT_CUDA_MALLOC
	EventTypeMemcpy           = 3  // EVENT_CUDA_MEMCPY
	EventTypeGraphLaunch      = 4  // EVENT_CUDA_GRAPH_LAUNCH
	EventTypeKernelLaunchDone = 5  // EVENT_CUDA_KERNEL_LAUNCH_DONE
	EventTypeSync             = 6  // EVENT_CUDA_SYNC
	EventTypeFree             = 7  // EVENT_CUDA_FREE
	EventTypeMemset           = 8  // EVENT_CUDA_MEMSET
	EventTypePeerCopy         = 9  // EVENT_CUDA_PEER_COPY
	EventTypeError            = 10 // EVENT_CUDA_ERROR
	EventTypeHamiOOM          = 11 // EVENT_HAMI_OOM
	EventTypeHamiThrottle     = 12 // EVENT_HAMI_THROTTLE

	// Memory kind constants (mirror C #defines)
	MemKindDevice  = 1
	MemKindHost    = 2
	MemKindManaged = 3
	MemKindPool    = 4

	// Sync kind constants (mirror C #defines)
	SyncKindStream = 1
	SyncKindDevice = 2
	SyncKindEvent  = 3

	// Function ID constants for error events (mirror C #defines)
	CudaFuncLaunch        = 1
	CudaFuncCoopLaunch    = 2
	CudaFuncMalloc        = 3
	CudaFuncManagedMalloc = 4
	CudaFuncHostMalloc    = 5
	CudaFuncHostAlloc     = 6
	CudaFuncAsyncMalloc   = 7
	CudaFuncSyncStream    = 8
	CudaFuncSyncDevice    = 9
	CudaFuncSyncEvent     = 10
)

type pidKey struct {
	Pid int32
	Ns  uint32
}

type (
	GPUCudaKernelLaunchInfo     BpfCudaKernelLaunchT
	GPUCudaMallocInfo           BpfCudaMallocT
	GPUCudaMemcpyInfo           BpfCudaMemcpyT
	GPUCudaGraphLaunchInfo      BpfCudaGraphLaunchT
	GPUCudaSyncInfo             BpfCudaSyncT
	GPUCudaFreeInfo             BpfCudaFreeT
	GPUCudaMemsetInfo           BpfCudaMemsetT
	GPUCudaPeerCopyInfo         BpfCudaPeerCopyT
	GPUCudaKernelLaunchDoneInfo BpfCudaKernelLaunchDoneT
	GPUCudaErrorInfo            BpfCudaErrorT
	GPUHamiOOMInfo              BpfHamiOomT
	GPUHamiThrottleInfo         BpfHamiThrottleT
)

// TODO: We have a way to bring ELF file information to this Tracer struct
// via the newNonGoTracersGroup / newNonGoTracersGroupUProbes functions. Now,
// we need to figure out how to pass it to the SharedRingbuf.. not sure if thats
// possible
type Tracer struct {
	pidsFilter       ebpfcommon.ServiceFilter
	cfg              *obi.Config
	metrics          imetrics.Reporter
	bpfObjects       BpfObjects
	closers          []io.Closer
	log              *slog.Logger
	instrumentedLibs ebpfcommon.InstrumentedLibsT
	libsMux          sync.Mutex
	pidMap           map[pidKey]uint64
}

func New(pidFilter ebpfcommon.ServiceFilter, cfg *obi.Config, metrics imetrics.Reporter) *Tracer {
	log := slog.With("component", "gpuevent.Tracer")

	log.Info("enabling CUDA kernel instrumentation")

	return &Tracer{
		log:              log,
		cfg:              cfg,
		metrics:          metrics,
		pidsFilter:       pidFilter,
		instrumentedLibs: make(ebpfcommon.InstrumentedLibsT),
		libsMux:          sync.Mutex{},
		pidMap:           map[pidKey]uint64{},
	}
}

func (p *Tracer) AllowPID(pid app.PID, ns uint32, svc *svc.Attrs) {
	p.pidsFilter.AllowPID(pid, ns, svc, ebpfcommon.PIDTypeKProbes)
}

func (p *Tracer) BlockPID(pid app.PID, ns uint32) {
	p.pidsFilter.BlockPID(pid, ns)
}

func (p *Tracer) LoadSpecs() ([]*ebpfcommon.SpecBundle, error) {
	spec, err := LoadBpf()
	if err != nil {
		return nil, err
	}

	return []*ebpfcommon.SpecBundle{{Spec: spec, Objects: &p.bpfObjects, Constants: p.constants()}}, nil
}

func (p *Tracer) constants() map[string]any {
	// The eBPF side does some basic filtering of events that do not belong to
	// processes which we monitor. We filter more accurately in the userspace, but
	// for performance reasons we enable the PID based filtering in eBPF.
	filterPids := int32(1)
	if p.cfg.Discovery.BPFPidFilterOff {
		filterPids = int32(0)
	}

	return map[string]any{
		"filter_pids": filterPids,
		"g_bpf_debug": p.cfg.EBPF.BpfDebug,
	}
}

func (p *Tracer) RegisterOffsets(_ *exec.FileInfo, _ *goexec.Offsets) {}

func (p *Tracer) ProcessBinary(_ *exec.FileInfo) {}

func (p *Tracer) AddCloser(c ...io.Closer) {
	p.closers = append(p.closers, c...)
}

func (p *Tracer) GoProbes() map[string][]*ebpfcommon.ProbeDesc {
	return nil
}

func (p *Tracer) KProbes() map[string]ebpfcommon.ProbeDesc {
	return nil
}

func (p *Tracer) Tracepoints() map[string]ebpfcommon.ProbeDesc {
	return nil
}

func (p *Tracer) UProbes() map[string]map[string][]*ebpfcommon.ProbeDesc {
	return map[string]map[string][]*ebpfcommon.ProbeDesc{
		// libvgpu.so probes — only-HAMi mode. Required:false so MIG clusters are unaffected.
		"libvgpu.so": {
			"cuLaunchKernel": {{
				Start:    p.bpfObjects.ObiHamiCuLaunch,
				Required: false,
			}},
			"cuLaunchCooperativeKernel": {{
				Start:    p.bpfObjects.ObiHamiCuCoopLaunch,
				Required: false,
			}},
			"cuMemAlloc_v2": {{
				End:      p.bpfObjects.ObiHamiCuMemAllocExit,
				Required: false,
			}},
			"cuMemAllocManaged": {{
				End:      p.bpfObjects.ObiHamiCuMemAllocManagedExit,
				Required: false,
			}},
			"cuMemAllocHost_v2": {{
				End:      p.bpfObjects.ObiHamiCuMemAllocHostExit,
				Required: false,
			}},
			"cuMemHostAlloc": {{
				End:      p.bpfObjects.ObiHamiCuMemHostAllocExit,
				Required: false,
			}},
			"cuMemAllocAsync": {{
				End:      p.bpfObjects.ObiHamiCuMemAllocAsyncExit,
				Required: false,
			}},
		},
		"libcuda.so": {
			// Kernel launches (entry emits grid/block info; exit emits duration)
			"cuLaunchKernel": {{
				Start: p.bpfObjects.ObiCuLaunch,
				End:   p.bpfObjects.ObiCuLaunchExit,
			}},
			"cuLaunchCooperativeKernel": {{
				Start: p.bpfObjects.ObiCuCoopLaunch,
				End:   p.bpfObjects.ObiCuCoopLaunchExit,
			}},
			// Graph launches
			"cuGraphLaunch": {{
				Start: p.bpfObjects.ObiCuGraphLaunch,
			}},
			// Device memory alloc/free (entry + exit for size tracking)
			"cuMemAlloc_v2": {{
				Start: p.bpfObjects.ObiCuMemAlloc,
				End:   p.bpfObjects.ObiCuMemAllocExit,
			}},
			"cuMemAllocManaged": {{
				Start: p.bpfObjects.ObiCuMemAllocManaged,
				End:   p.bpfObjects.ObiCuMemAllocManagedExit,
			}},
			"cuMemAllocHost_v2": {{
				Start: p.bpfObjects.ObiCuMemAllocHost,
				End:   p.bpfObjects.ObiCuMemAllocHostExit,
			}},
			"cuMemHostAlloc": {{
				Start: p.bpfObjects.ObiCuMemHostAlloc,
				End:   p.bpfObjects.ObiCuMemHostAllocExit,
			}},
			"cuMemAllocAsync": {{
				Start: p.bpfObjects.ObiCuMemAllocAsync,
				End:   p.bpfObjects.ObiCuMemAllocAsyncExit,
			}},
			// Free probes
			"cuMemFree_v2": {{
				Start: p.bpfObjects.ObiCuMemFree,
			}},
			"cuMemFreeHost": {{
				Start: p.bpfObjects.ObiCuMemFreeHost,
			}},
			"cuMemFreeAsync": {{
				Start: p.bpfObjects.ObiCuMemFreeAsync,
			}},
			// Memcpy probes (direction-tagged by function name)
			"cuMemcpyHtoDAsync_v2": {{
				Start: p.bpfObjects.ObiCuMemcpyHtod,
			}},
			"cuMemcpyDtoHAsync_v2": {{
				Start: p.bpfObjects.ObiCuMemcpyDtoh,
			}},
			"cuMemcpyDtoDAsync_v2": {{
				Start: p.bpfObjects.ObiCuMemcpyDtod,
			}},
			"cuMemcpyPeer": {{
				Start: p.bpfObjects.ObiCuMemcpyPeer,
			}},
			"cuMemcpyPeerAsync": {{
				Start: p.bpfObjects.ObiCuMemcpyPeerAsync,
			}},
			// Memset probes
			"cuMemsetD8_v2": {{
				Start: p.bpfObjects.ObiCuMemset,
			}},
			"cuMemsetD8Async": {{
				Start: p.bpfObjects.ObiCuMemsetAsync,
			}},
			// Synchronize probes (entry + exit for duration)
			"cuStreamSynchronize": {{
				Start: p.bpfObjects.ObiCuStreamSync,
				End:   p.bpfObjects.ObiCuStreamSyncExit,
			}},
			"cuCtxSynchronize": {{
				Start: p.bpfObjects.ObiCuCtxSync,
				End:   p.bpfObjects.ObiCuCtxSyncExit,
			}},
			"cuEventSynchronize": {{
				Start: p.bpfObjects.ObiCuEventSync,
				End:   p.bpfObjects.ObiCuEventSyncExit,
			}},
		},
	}
}

func (p *Tracer) SetupTailCalls() {}

func (p *Tracer) SocketFilters() []*ebpf.Program { return nil }

func (p *Tracer) SockMsgs() []ebpfcommon.SockMsg { return nil }

func (p *Tracer) SockOps() []ebpfcommon.SockOps { return nil }

func (p *Tracer) Iters() []*ebpfcommon.Iter { return nil }

func (p *Tracer) Tracing() []*ebpfcommon.Tracing { return nil }

func (p *Tracer) RecordInstrumentedLib(id uint64, closers []io.Closer) {
	p.libsMux.Lock()
	defer p.libsMux.Unlock()

	module := p.instrumentedLibs.AddRef(id)

	if len(closers) > 0 {
		module.Closers = append(module.Closers, closers...)
	}

	p.log.Debug("Recorded instrumented Lib", "ino", id, "module", module)
}

func (p *Tracer) AddInstrumentedLibRef(id uint64) {
	p.RecordInstrumentedLib(id, nil)
}

func (p *Tracer) UnlinkInstrumentedLib(id uint64) {
	p.libsMux.Lock()
	defer p.libsMux.Unlock()

	module, err := p.instrumentedLibs.RemoveRef(id)

	p.log.Debug("Unlinking instrumented lib - before state", "ino", id, "module", module)

	if err != nil {
		p.log.Debug("Error unlinking instrumented lib", "ino", id, "error", err)
	}
}

func (p *Tracer) AlreadyInstrumentedLib(id uint64) bool {
	p.libsMux.Lock()
	defer p.libsMux.Unlock()

	module := p.instrumentedLibs.Find(id)

	p.log.Debug("checking already instrumented Lib", "ino", id, "module", module)
	return module != nil
}

func (p *Tracer) Run(ctx context.Context, ebpfEventContext *ebpfcommon.EBPFEventContext, eventsChan *msg.Queue[[]request.Span]) {
	ebpfcommon.ForwardRingbuf(
		&p.cfg.EBPF,
		p.bpfObjects.GpuEvents,
		p.processCudaEvent,
		ebpfEventContext.CommonPIDsFilter.Filter,
		p.log,
		p.metrics,
		append(p.closers, &p.bpfObjects)...,
	)(ctx, eventsChan)
}

func (p *Tracer) processCudaEvent(record *ringbuf.Record) (request.Span, bool, error) {
	if len(record.RawSample) == 0 {
		return request.Span{}, true, errors.New("invalid ringbuffer record size")
	}

	eventType := record.RawSample[0]

	switch eventType {
	case EventTypeKernelLaunch:
		return p.readGPUKernelLaunchIntoSpan(record)
	case EventTypeGraphLaunch:
		return p.readGPUGraphLaunchIntoSpan(record)
	case EventTypeMalloc:
		return p.readGPUMallocIntoSpan(record)
	case EventTypeMemcpy:
		return p.readGPUMemcpyIntoSpan(record)
	case EventTypeSync:
		return p.readGPUSyncIntoSpan(record)
	case EventTypeFree:
		return p.readGPUFreeIntoSpan(record)
	case EventTypeMemset:
		return p.readGPUMemsetIntoSpan(record)
	case EventTypePeerCopy:
		return p.readGPUPeerCopyIntoSpan(record)
	case EventTypeKernelLaunchDone:
		return p.readGPUKernelLaunchDoneIntoSpan(record)
	case EventTypeError:
		return p.readGPUErrorIntoSpan(record)
	case EventTypeHamiOOM:
		return p.readGPUHamiOOMIntoSpan(record)
	case EventTypeHamiThrottle:
		return p.readGPUHamiThrottleIntoSpan(record)
	default:
		p.log.Error("unknown cuda event", "type", eventType)
	}

	return request.Span{}, true, nil
}

func (p *Tracer) readGPUMallocIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaMallocInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Malloc", "event", event)

	return request.Span{
		Type:          request.EventTypeGPUCudaMalloc,
		ContentLength: event.Size,
		SubType:       int(event.MemKind),
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}, false, nil
}

func (p *Tracer) readGPUMemcpyIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaMemcpyInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Memcpy", "event", event)

	return request.Span{
		Type:          request.EventTypeGPUCudaMemcpy,
		ContentLength: event.Size,
		SubType:       int(event.Direction),
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}, false, nil
}

func (p *Tracer) readGPUKernelLaunchIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaKernelLaunchInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Kernel Launch", "event", event)

	return request.Span{
		Type:          request.EventTypeGPUCudaKernelLaunch,
		ContentLength: int64(event.GridX * event.GridY * event.GridZ),
		SubType:       int(event.BlockX * event.BlockY * event.BlockZ),
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}, false, nil
}

func (p *Tracer) readGPUGraphLaunchIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaGraphLaunchInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Graph Launch", "event", event)

	return request.Span{
		Type: request.EventTypeGPUCudaGraphLaunch,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}, false, nil
}

// readGPUSyncIntoSpan decodes a synchronize event and stores timing so Timings() yields duration.
// RequestStart and End are set to BPF monotonic timestamps so that
// End.Sub(RequestStart) == duration_ns (the actual GPU-wait interval).
func (p *Tracer) readGPUSyncIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaSyncInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Sync", "kind", event.SyncKind, "duration_ns", event.DurationNs)

	// Map BPF monotonic ns to Go monotonic ns.
	// monotime.Now() and bpf_ktime_get_ns() share CLOCK_MONOTONIC.
	monoNow := int64(monotime.Now())
	bpfNow := int64(event.EntryTs) + int64(event.DurationNs)
	delta := monoNow - bpfNow

	eventType := request.EventTypeGPUCudaStreamSync
	switch event.SyncKind {
	case SyncKindDevice:
		eventType = request.EventTypeGPUCudaDeviceSync
	case SyncKindEvent:
		eventType = request.EventTypeGPUCudaEventSync
	}

	return request.Span{
		Type:         eventType,
		RequestStart: int64(event.EntryTs) + delta,
		End:          int64(event.EntryTs) + int64(event.DurationNs) + delta,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}, false, nil
}

func (p *Tracer) readGPUFreeIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaFreeInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Free", "kind", event.MemKind, "size", event.Size)

	return request.Span{
		Type:          request.EventTypeGPUCudaFree,
		ContentLength: event.Size,
		SubType:       int(event.MemKind),
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}, false, nil
}

func (p *Tracer) readGPUMemsetIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaMemsetInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Memset", "async", event.IsAsync, "size", event.Size)

	return request.Span{
		Type:          request.EventTypeGPUCudaMemset,
		ContentLength: event.Size,
		SubType:       int(event.IsAsync),
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}, false, nil
}

func (p *Tracer) readGPUPeerCopyIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaPeerCopyInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Peer Copy", "src", event.SrcDevice, "dst", event.DstDevice, "size", event.Size)

	return request.Span{
		Type:          request.EventTypeGPUCudaPeerCopy,
		ContentLength: event.Size,
		// Pack src/dst device IDs: src in high 16 bits, dst in low 16 bits
		SubType: int(event.SrcDevice)<<16 | int(event.DstDevice)&0xFFFF,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}, false, nil
}

// readGPUKernelLaunchDoneIntoSpan decodes a kernel-launch-done event, providing
// host-side launch latency (time from cudaLaunchKernel entry to return).
func (p *Tracer) readGPUKernelLaunchDoneIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaKernelLaunchDoneInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Kernel Launch Done", "duration_ns", event.DurationNs, "retval", event.Retval)

	monoNow := int64(monotime.Now())
	bpfNow := int64(event.EntryTs) + int64(event.DurationNs)
	delta := monoNow - bpfNow

	return request.Span{
		Type:         request.EventTypeGPUCudaKernelLaunchDone,
		RequestStart: int64(event.EntryTs) + delta,
		End:          int64(event.EntryTs) + int64(event.DurationNs) + delta,
		SubType:      int(event.Retval),
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}, false, nil
}

// readGPUErrorIntoSpan decodes a CUDA error event (non-zero cudaError_t return value).
func (p *Tracer) readGPUErrorIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaErrorInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU CUDA Error", "func_id", event.FuncId, "error_code", event.ErrorCode)

	return request.Span{
		Type:          request.EventTypeGPUCudaError,
		SubType:       int(event.ErrorCode), // cudaError_t value
		ContentLength: int64(event.FuncId),  // CUDA_FUNC_* identifier → mapped to function name
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}, false, nil
}

// readGPUHamiOOMIntoSpan decodes a HAMi quota-OOM event from libvgpu.so.
// SubType packs both mem_kind and rc: (int(mem_kind) << 24) | int(rc).
// ContentLength carries the CUDA_FUNC_* identifier.
func (p *Tracer) readGPUHamiOOMIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUHamiOOMInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("HAMi OOM", "func_id", event.CudaFuncId, "mem_kind", event.MemKind, "rc", event.Rc)

	return request.Span{
		Type:          request.EventTypeGPUHamiOOM,
		SubType:       (int(event.MemKind) << 24) | int(event.Rc),
		ContentLength: int64(event.CudaFuncId),
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}, false, nil
}

// readGPUHamiThrottleIntoSpan decodes a HAMi compute-throttle event.
// RequestStart and End are set so that End-RequestStart == duration_ns.
func (p *Tracer) readGPUHamiThrottleIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUHamiThrottleInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("HAMi Throttle", "duration_ns", event.DurationNs)

	monoNow := int64(monotime.Now())
	// BPF measures duration from libvgpu entry to libcuda entry (both ktime).
	// We don't have entry_ts, so anchor End to now.
	end := monoNow
	start := end - int64(event.DurationNs)

	return request.Span{
		Type:         request.EventTypeGPUHamiThrottle,
		RequestStart: start,
		End:          end,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}, false, nil
}

func (p *Tracer) SetEventContext(_ *ebpfcommon.EBPFEventContext) {}

func (p *Tracer) Capabilities() ebpfcommon.TracerCapability { return 0 }

func (p *Tracer) Required() bool {
	return false
}
