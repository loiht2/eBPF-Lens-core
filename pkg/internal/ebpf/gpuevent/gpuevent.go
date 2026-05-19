// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpuevent // import "go.opentelemetry.io/obi/pkg/internal/ebpf/gpuevent"

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"strings"
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
	"go.opentelemetry.io/obi/pkg/internal/hami"
	"go.opentelemetry.io/obi/pkg/obi"
	"go.opentelemetry.io/obi/pkg/pipe/msg"
)

//go:generate $BPF2GO -cc $BPF_CLANG -cflags $BPF_CFLAGS -type cuda_kernel_launch_t -type cuda_graph_launch_t -type cuda_malloc_t -type cuda_memcpy_t -type cuda_sync_t -type cuda_free_t -type cuda_memset_t -type cuda_peer_copy_t -type cuda_kernel_launch_done_t -type cuda_error_t -type hami_oom_t -type hami_throttle_t -type cuda_event_elapsed_t -target amd64,arm64 Bpf ../../../../bpf/gpuevent/gpuevent.c -- -I../../../../bpf

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
	EventTypeEventElapsed     = 13 // EVENT_CUDA_EVENT_ELAPSED — GPU time from cuEventElapsedTime

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
	CudaFuncPoolMalloc    = 11
	CudaFuncHostRegister  = 12
	CudaFuncHostUnregister = 13
	CudaFuncEventElapsed  = 14
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
	GPUEventElapsedInfo         BpfCudaEventElapsedT
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
	// gpuUUIDCache caches the GPU UUID resolved from /proc/<pid>/environ per host PID.
	// Used as the MIG-mode fallback when the HAMi PIDIndex has no entry for this PID.
	// Entries are populated lazily on first event for each PID and evicted in BlockPID.
	gpuUUIDCache sync.Map // map[int32]string
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
	p.gpuUUIDCache.Delete(int32(pid))
}

// withGPUUUID enriches a span with the physical GPU UUID for its PID.
// Resolution order (all results are cached in gpuUUIDCache per host PID):
//  1. HAMi PIDIndex (device-plugin mode): populated by cache poller every ~1 s.
//     Only works when libvgpu.so records the real host PID as hostpid in the cache.
//  2. CUDA_DEVICE_MEMORY_SHARED_CACHE env var (HAMi DRA mode): read the cache file
//     pointed to by the env var and return sr.uuids[0].
//  3. NVIDIA_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES with MIG- prefix (only-MIG mode).
//
// Returns the span unchanged when all sources return empty.
func (p *Tracer) withGPUUUID(span request.Span) request.Span {
	hostPID := int32(span.Pid.HostPID)

	// 1. HAMi PIDIndex (fast path; populated by poller for device-plugin mode)
	if uuid := hami.DefaultPIDIndex().LookupUUID(hostPID); uuid != "" {
		span.GPUUuid = uuid
		return span
	}

	// 2+3. /proc/environ: try HAMi DRA cache file, then MIG env var (one read, cached)
	if v, ok := p.gpuUUIDCache.Load(hostPID); ok {
		span.GPUUuid = v.(string)
		return span
	}
	uuid := resolveGPUUUIDFromProc(hostPID)
	p.gpuUUIDCache.Store(hostPID, uuid)
	span.GPUUuid = uuid
	return span
}

// resolveGPUUUIDFromProc reads /proc/<pid>/environ once and resolves the GPU UUID via:
//  a. CUDA_DEVICE_MEMORY_SHARED_CACHE (HAMi DRA): open the cache file, return sr.uuids[0].
//  b. CUDA_VISIBLE_DEVICES / NVIDIA_VISIBLE_DEVICES with "MIG-" prefix (only-MIG).
func resolveGPUUUIDFromProc(pid int32) string {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/environ", pid))
	if err != nil {
		return ""
	}
	var cachePath, visDevices string
	for _, entry := range bytes.Split(data, []byte{0}) {
		kv := bytes.SplitN(entry, []byte{'='}, 2)
		if len(kv) != 2 {
			continue
		}
		switch string(kv[0]) {
		case "CUDA_DEVICE_MEMORY_SHARED_CACHE":
			cachePath = strings.TrimSpace(string(kv[1]))
		case "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES":
			val := strings.TrimSpace(string(kv[1]))
			if val != "" && val != "void" && val != "NoDevFiles" {
				visDevices = val
			}
		}
	}

	// HAMi DRA: read UUID from the cache file (most reliable source)
	if cachePath != "" {
		if uuid := readUUIDFromCacheFile(cachePath); uuid != "" {
			return uuid
		}
	}

	// only-MIG: NVIDIA DRA injects "MIG-<uuid>" into NVIDIA_VISIBLE_DEVICES.
	if visDevices != "" {
		return strings.TrimPrefix(visDevices, "MIG-")
	}
	return ""
}

// readUUIDFromCacheFile opens a HAMi cudevshr.cache file and returns the GPU UUID
// for device 0 (sr.uuids[0]).
//
// The agent container may not have the HAMi cache directory mounted as a volume.
// With hostPID=true + privileged=true the host filesystem is reachable via
// /proc/1/root/<path>, so we try both the direct path and the /proc/1/root prefix.
func readUUIDFromCacheFile(path string) string {
	for _, prefix := range []string{"", "/proc/1/root"} {
		sr, err := hami.OpenSharedRegion(prefix + path)
		if err != nil {
			continue
		}
		sample := sr.Snapshot("", "")
		sr.Close()
		if len(sample.Devices) > 0 && sample.Devices[0].UUID != "" {
			return sample.Devices[0].UUID
		}
	}
	return ""
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
			// cuLaunchKernelEx — CUDA 11.4+; required by PyTorch 2.4+ (torch.compile / Inductor)
			// and NCCL 2.20+ collectives. Required:false so older libcuda.so without
			// this symbol does not break attachment on legacy clusters.
			"cuLaunchKernelEx": {{
				Start:    p.bpfObjects.ObiCuLaunchEx,
				End:      p.bpfObjects.ObiCuLaunchExExit,
				Required: false,
			}},
			// Graph launches (entry emits the launch event; exit is error-only)
			"cuGraphLaunch": {{
				Start: p.bpfObjects.ObiCuGraphLaunch,
				End:   p.bpfObjects.ObiCuGraphLaunchExit,
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
			// cuMemAllocFromPoolAsync — required by PyTorch 2.1+ custom pool, NCCL 2.19+ scratch.
			// Required:false so older libcuda.so (no symbol) does not break attachment.
			"cuMemAllocFromPoolAsync": {{
				Start:    p.bpfObjects.ObiCuMemAllocFromPool,
				End:      p.bpfObjects.ObiCuMemAllocFromPoolExit,
				Required: false,
			}},
			// cuMemHostRegister — pin existing host memory (NCCL staging buffers, PyTorch pin_memory=True).
			"cuMemHostRegister": {{
				Start:    p.bpfObjects.ObiCuMemHostRegister,
				End:      p.bpfObjects.ObiCuMemHostRegisterExit,
				Required: false,
			}},
			"cuMemHostUnregister": {{
				Start:    p.bpfObjects.ObiCuMemHostUnregister,
				End:      p.bpfObjects.ObiCuMemHostUnregisterExit,
				Required: false,
			}},
			// cuEventElapsedTime — capture GPU time between two CUevents. Required:false because
			// older inference workloads do not use event timing; we still want clusters without the
			// symbol to attach cleanly.
			"cuEventElapsedTime": {{
				Start:    p.bpfObjects.ObiCuEventElapsed,
				End:      p.bpfObjects.ObiCuEventElapsedExit,
				Required: false,
			}},
			// Free probes — entry stashes inflight state, exit emits event only on rc==CUDA_SUCCESS.
			"cuMemFree_v2": {{
				Start: p.bpfObjects.ObiCuMemFree,
				End:   p.bpfObjects.ObiCuMemFreeExit,
			}},
			"cuMemFreeHost": {{
				Start: p.bpfObjects.ObiCuMemFreeHost,
				End:   p.bpfObjects.ObiCuMemFreeHostExit,
			}},
			"cuMemFreeAsync": {{
				Start: p.bpfObjects.ObiCuMemFreeAsync,
				End:   p.bpfObjects.ObiCuMemFreeAsyncExit,
			}},
			// Memcpy probes — async variants only. Entry stashes (size, direction, stream);
			// exit emits only on rc==CUDA_SUCCESS (submission accepted by driver).
			"cuMemcpyHtoDAsync_v2": {{
				Start: p.bpfObjects.ObiCuMemcpyHtod,
				End:   p.bpfObjects.ObiCuMemcpyHtodExit,
			}},
			"cuMemcpyDtoHAsync_v2": {{
				Start: p.bpfObjects.ObiCuMemcpyDtoh,
				End:   p.bpfObjects.ObiCuMemcpyDtohExit,
			}},
			"cuMemcpyDtoDAsync_v2": {{
				Start: p.bpfObjects.ObiCuMemcpyDtod,
				End:   p.bpfObjects.ObiCuMemcpyDtodExit,
			}},
			"cuMemcpyPeer": {{
				Start: p.bpfObjects.ObiCuMemcpyPeer,
				End:   p.bpfObjects.ObiCuMemcpyPeerExit,
			}},
			"cuMemcpyPeerAsync": {{
				Start: p.bpfObjects.ObiCuMemcpyPeerAsync,
				End:   p.bpfObjects.ObiCuMemcpyPeerAsyncExit,
			}},
			// Memset probes — entry stashes (size, is_async, stream); exit emits only on rc==CUDA_SUCCESS.
			"cuMemsetD8_v2": {{
				Start: p.bpfObjects.ObiCuMemset,
				End:   p.bpfObjects.ObiCuMemsetExit,
			}},
			"cuMemsetD8Async": {{
				Start: p.bpfObjects.ObiCuMemsetAsync,
				End:   p.bpfObjects.ObiCuMemsetAsyncExit,
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
	case EventTypeEventElapsed:
		return p.readGPUEventElapsedIntoSpan(record)
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

	return p.withGPUUUID(request.Span{
		Type:          request.EventTypeGPUCudaMalloc,
		ContentLength: event.Size,
		SubType:       int(event.MemKind),
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}), false, nil
}

func (p *Tracer) readGPUMemcpyIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaMemcpyInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Memcpy", "event", event)

	return p.withGPUUUID(request.Span{
		Type:            request.EventTypeGPUCudaMemcpy,
		ContentLength:   event.Size,
		SubType:         int(event.Direction),
		GPUStreamHandle: event.StreamHandle,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}), false, nil
}

func (p *Tracer) readGPUKernelLaunchIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaKernelLaunchInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Kernel Launch", "event", event)

	// Cast each dimension to uint64 before multiplication to prevent int32
	// overflow when grid x*y*z exceeds 2^31 (e.g. large GEMM kernels).
	gridTotal := uint64(uint32(event.GridX)) * uint64(uint32(event.GridY)) * uint64(uint32(event.GridZ))
	blockTotal := uint64(uint32(event.BlockX)) * uint64(uint32(event.BlockY)) * uint64(uint32(event.BlockZ))

	return p.withGPUUUID(request.Span{
		Type:              request.EventTypeGPUCudaKernelLaunch,
		ContentLength:     int64(gridTotal),
		SubType:           int(blockTotal),
		GPUSharedMemBytes: event.SharedMemBytes,
		GPUStreamHandle:   event.StreamHandle,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}), false, nil
}

func (p *Tracer) readGPUGraphLaunchIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaGraphLaunchInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Graph Launch", "event", event)

	return p.withGPUUUID(request.Span{
		Type: request.EventTypeGPUCudaGraphLaunch,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}), false, nil
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

	// For stream/event sync, the handle is the CUstream/CUevent argument.
	// For device sync (cuCtxSynchronize), handle is 0 (current context not
	// available from BPF). Surface as GPUStreamHandle for stream sync, and
	// GPUEventHandle for event sync.
	span := request.Span{
		Type:         eventType,
		RequestStart: int64(event.EntryTs) + delta,
		End:          int64(event.EntryTs) + int64(event.DurationNs) + delta,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}
	switch event.SyncKind {
	case SyncKindEvent:
		span.GPUEventHandle = event.Handle
	default:
		span.GPUStreamHandle = event.Handle
	}
	return p.withGPUUUID(span), false, nil
}

func (p *Tracer) readGPUFreeIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaFreeInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Free", "kind", event.MemKind, "size", event.Size)

	return p.withGPUUUID(request.Span{
		Type:            request.EventTypeGPUCudaFree,
		ContentLength:   event.Size,
		SubType:         int(event.MemKind),
		GPUStreamHandle: event.StreamHandle,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}), false, nil
}

func (p *Tracer) readGPUMemsetIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaMemsetInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Memset", "async", event.IsAsync, "size", event.Size)

	return p.withGPUUUID(request.Span{
		Type:            request.EventTypeGPUCudaMemset,
		ContentLength:   event.Size,
		SubType:         int(event.IsAsync),
		GPUStreamHandle: event.StreamHandle,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}), false, nil
}

func (p *Tracer) readGPUPeerCopyIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaPeerCopyInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU Peer Copy", "src", event.SrcDevice, "dst", event.DstDevice, "size", event.Size)

	return p.withGPUUUID(request.Span{
		Type:          request.EventTypeGPUCudaPeerCopy,
		ContentLength: event.Size,
		// Pack src/dst device IDs: src in high 16 bits, dst in low 16 bits
		SubType:         int(event.SrcDevice)<<16 | int(event.DstDevice)&0xFFFF,
		GPUStreamHandle: event.StreamHandle,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}), false, nil
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

	return p.withGPUUUID(request.Span{
		Type:         request.EventTypeGPUCudaKernelLaunchDone,
		RequestStart: int64(event.EntryTs) + delta,
		End:          int64(event.EntryTs) + int64(event.DurationNs) + delta,
		SubType:      int(event.Retval),
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}), false, nil
}

// readGPUErrorIntoSpan decodes a CUDA error event (non-zero cudaError_t return value).
func (p *Tracer) readGPUErrorIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUCudaErrorInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	p.log.Debug("GPU CUDA Error", "func_id", event.FuncId, "error_code", event.ErrorCode)

	return p.withGPUUUID(request.Span{
		Type:          request.EventTypeGPUCudaError,
		SubType:       int(event.ErrorCode), // cudaError_t value
		ContentLength: int64(event.FuncId),  // CUDA_FUNC_* identifier → mapped to function name
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}), false, nil
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

	// Pack mem_kind (high byte) and rc (low 24 bits) into SubType.
	// Use uint32 + explicit 24-bit mask on rc to prevent sign-extension when
	// HAMi returns a negative int32 (e.g. cuMemAllocAsync returns -1, which
	// previously corrupted the mem_kind bits via Go's sign-extending int cast).
	rc24 := int(uint32(event.Rc)) & 0xFFFFFF
	return p.withGPUUUID(request.Span{
		Type:          request.EventTypeGPUHamiOOM,
		SubType:       (int(event.MemKind) << 24) | rc24,
		ContentLength: int64(event.CudaFuncId),
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}), false, nil
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

	return p.withGPUUUID(request.Span{
		Type:         request.EventTypeGPUHamiThrottle,
		RequestStart: start,
		End:          end,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}), false, nil
}

// readGPUEventElapsedIntoSpan decodes a cuEventElapsedTime success event.
// The BPF probe captured the IEEE-754 float32 milliseconds bit pattern; Go
// reinterprets it. Stream/event handles are stored on the span (internal)
// but NOT exposed as Prometheus labels — they are opaque high-cardinality
// pointers.
//
// RequestStart and End are anchored to "now" so End-RequestStart equals the
// elapsed GPU duration in nanoseconds (mirrors the cuda_sync pattern so the
// existing Timings() machinery works for histogram observation).
func (p *Tracer) readGPUEventElapsedIntoSpan(record *ringbuf.Record) (request.Span, bool, error) {
	event, err := ebpfcommon.ReinterpretCast[GPUEventElapsedInfo](record.RawSample)
	if err != nil {
		return request.Span{}, true, err
	}

	// Decode IEEE-754 float32 milliseconds → nanoseconds (int64).
	// Defensive: clamp negative/NaN/inf to 0; cuEventElapsedTime should always return ms >= 0.
	elapsedMs := math.Float32frombits(event.ElapsedMsBits)
	var elapsedNs int64
	if elapsedMs > 0 && !math.IsNaN(float64(elapsedMs)) && !math.IsInf(float64(elapsedMs), 0) {
		elapsedNs = int64(float64(elapsedMs) * 1e6)
	}

	p.log.Debug("GPU Event Elapsed", "elapsed_ms", elapsedMs, "h_start", event.H_start, "h_end", event.H_end)

	end := int64(monotime.Now())
	start := end - elapsedNs
	return p.withGPUUUID(request.Span{
		Type:           request.EventTypeGPUCudaEventElapsed,
		RequestStart:   start,
		End:            end,
		GPUEventHandle: event.H_start,
		Pid: request.PidInfo{
			HostPID:   app.PID(event.PidInfo.HostPid),
			UserPID:   app.PID(event.PidInfo.UserPid),
			Namespace: event.PidInfo.Ns,
		},
	}), false, nil
}

func (p *Tracer) SetEventContext(_ *ebpfcommon.EBPFEventContext) {}

func (p *Tracer) Capabilities() ebpfcommon.TracerCapability { return 0 }

func (p *Tracer) Required() bool {
	return false
}
