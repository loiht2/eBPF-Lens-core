// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build obi_bpf_ignore
// Source: https://github.com/facebookincubator/strobelight/blob/5d84bcfdd9abccc615b45a390bfd7bba7097dc51/strobelight/src/profilers/gpuevent_snoop/bpf/gpuevent_snoop.bpf.c
// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <bpfcore/vmlinux.h>
#include <bpfcore/bpf_helpers.h>
#include <bpfcore/bpf_tracing.h>

#include <gpuevent/cuda.h>
#include <gpuevent/gpu_ringbuf.h>

#include <logger/bpf_dbg.h>

#include <pid/pid.h>

// Force bpf2go to generate Go types for all ring-buffer event structs
const cuda_kernel_launch_t      *unused_gpu  __attribute__((unused));
const cuda_malloc_t             *unused_gpu1 __attribute__((unused));
const cuda_memcpy_t             *unused_gpu2 __attribute__((unused));
const cuda_graph_launch_t       *unused_gpu3 __attribute__((unused));
const cuda_sync_t               *unused_gpu4 __attribute__((unused));
const cuda_free_t               *unused_gpu5 __attribute__((unused));
const cuda_memset_t             *unused_gpu6 __attribute__((unused));
const cuda_peer_copy_t          *unused_gpu7 __attribute__((unused));
const cuda_kernel_launch_done_t *unused_gpu8 __attribute__((unused));
const cuda_error_t              *unused_gpu9 __attribute__((unused));
const hami_oom_t                *unused_gpu10 __attribute__((unused));
const hami_throttle_t           *unused_gpu11 __attribute__((unused));
const cuda_event_elapsed_t      *unused_gpu12 __attribute__((unused));

enum {
    k_event_kernel_launch      = 1,
    k_event_malloc             = 2,
    k_event_memcpy             = 3,
    k_event_graph_launch       = 4,
    k_event_kernel_launch_done = 5,
    k_event_sync               = 6,
    k_event_free               = 7,
    k_event_memset             = 8,
    k_event_peer_copy          = 9,
    k_event_error              = 10,
    k_event_hami_oom           = 11,
    k_event_hami_throttle      = 12,
    k_event_event_elapsed      = 13, // cuEventElapsedTime — GPU time between two recorded CUevents
};

// ──────────────────────────── BPF maps ────────────────────────────

// Maps pid_tgid → entry timestamp for in-flight sync calls
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key,   u64);
    __type(value, cuda_ongoing_sync_t);
} gpu_sync_start SEC(".maps");

// Maps pid_tgid → {devptr_addr, size, mem_kind} for in-flight alloc calls
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key,   u64);
    __type(value, cuda_ongoing_alloc_t);
} gpu_ongoing_alloc SEC(".maps");

// Maps (ptr, tgid) → {size, mem_kind} so cuMemFree can report bytes freed.
// Composite key prevents cross-process collisions: two pods can hold the same
// virtual pointer value in their separate address spaces.
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 131072);
    __type(key,   cuda_alloc_key_t);
    __type(value, cuda_alloc_record_t);
} gpu_alloc_sizes SEC(".maps");

// Maps pid_tgid → entry timestamp for in-flight kernel launch calls
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key,   u64);
    __type(value, u64);
} gpu_launch_start SEC(".maps");

// Maps pid_tgid → libvgpu.so:cuLaunchKernel entry timestamp (only-HAMi mode).
// Written by hami.c entry probes; read+deleted in cu_launch_entry_impl to compute
// HAMi rate_limiter stall duration. Always empty in only-MIG mode (libvgpu absent).
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 65536);
    __type(key,   u64);
    __type(value, u64);
} hami_launch_entry SEC(".maps");

// In-flight per-PID maps used by entry+exit probes to defer ringbuf emission
// until the call's return code is known. Entries are removed unconditionally
// on uretprobe; the event is only emitted on rc==CUDA_SUCCESS.
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key,   u64);
    __type(value, cuda_ongoing_free_t);
} gpu_ongoing_free SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key,   u64);
    __type(value, cuda_ongoing_memcpy_t);
} gpu_ongoing_memcpy SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key,   u64);
    __type(value, cuda_ongoing_memset_t);
} gpu_ongoing_memset SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key,   u64);
    __type(value, cuda_ongoing_peer_copy_t);
} gpu_ongoing_peer_copy SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key,   u64);
    __type(value, cuda_ongoing_host_reg_t);
} gpu_ongoing_host_reg SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key,   u64);
    __type(value, cuda_ongoing_event_elapsed_t);
} gpu_ongoing_event_elapsed SEC(".maps");

// ──────────────────────── Shared helpers ──────────────────────────

static __always_inline void cuda_error_impl(struct pt_regs *ctx, u8 func_id, s32 error_code) {
    const u64 id = bpf_get_current_pid_tgid();
    cuda_error_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        return;
    }
    e->flags      = k_event_error;
    e->func_id    = func_id;
    task_pid(&e->pid_info);
    e->error_code = error_code;
    bpf_ringbuf_submit(e, 0);
    (void)id;
}

static __always_inline int cuda_sync_entry_impl(void *ctx, u64 handle) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    cuda_ongoing_sync_t inflight = {
        .entry_ts = bpf_ktime_get_ns(),
        .handle   = handle,
    };
    bpf_map_update_elem(&gpu_sync_start, &id, &inflight, BPF_ANY);
    return 0;
}

static __always_inline int cuda_sync_exit_impl(struct pt_regs *ctx, u8 sync_kind, u8 func_id) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    cuda_ongoing_sync_t *inflight = bpf_map_lookup_elem(&gpu_sync_start, &id);
    if (!inflight) {
        return 0;
    }
    u64 now = bpf_ktime_get_ns();
    s32 rc = (s32)(long)PT_REGS_RC(ctx);
    u64 duration = now - inflight->entry_ts;
    u64 saved_entry = inflight->entry_ts;
    u64 saved_handle = inflight->handle;
    bpf_map_delete_elem(&gpu_sync_start, &id);

    cuda_sync_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        return 0;
    }
    e->flags       = k_event_sync;
    e->sync_kind   = sync_kind;
    task_pid(&e->pid_info);
    e->entry_ts    = saved_entry;
    e->duration_ns = duration;
    e->retval      = rc;
    e->handle      = saved_handle;
    bpf_ringbuf_submit(e, 0);

    if (rc != 0) {
        cuda_error_impl(ctx, func_id, rc);
    }
    return 0;
}

static __always_inline int cuda_alloc_entry_impl(void *ctx, void **devptr_arg, size_t size, u8 mem_kind) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    // Record devptr_arg address + size in ongoing map for exit probe.
    // Event is emitted at exit so failed allocations are not counted in bytes.
    cuda_ongoing_alloc_t rec = {
        .devptr_addr = (u64)devptr_arg,
        .size        = (s64)size,
        .mem_kind    = mem_kind,
    };
    bpf_map_update_elem(&gpu_ongoing_alloc, &id, &rec, BPF_ANY);
    return 0;
}

static __always_inline int cuda_alloc_exit_impl(struct pt_regs *ctx, u8 func_id) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    cuda_ongoing_alloc_t *rec = bpf_map_lookup_elem(&gpu_ongoing_alloc, &id);
    if (!rec) {
        return 0;
    }
    s32 rc = (s32)(long)PT_REGS_RC(ctx);
    // Read the allocated pointer from userspace (the function wrote to *devptr_addr)
    u64 ptr_val = 0;
    bpf_probe_read_user(&ptr_val, sizeof(ptr_val), (void *)rec->devptr_addr);
    if (rc == 0) {
        // Emit malloc event only for successful allocations
        cuda_malloc_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
        if (e) {
            e->flags = k_event_malloc;
            e->mem_kind = rec->mem_kind;
            task_pid(&e->pid_info);
            e->size = rec->size;
            bpf_ringbuf_submit(e, 0);
        }
        if (ptr_val != 0) {
            cuda_alloc_record_t alloc_rec = {
                .size     = rec->size,
                .mem_kind = rec->mem_kind,
            };
            cuda_alloc_key_t key = {
                .ptr  = ptr_val,
                .tgid = (u32)(id >> 32),
                ._pad = 0,
            };
            bpf_map_update_elem(&gpu_alloc_sizes, &key, &alloc_rec, BPF_ANY);
        }
    } else {
        cuda_error_impl(ctx, func_id, rc);
    }
    bpf_map_delete_elem(&gpu_ongoing_alloc, &id);
    return 0;
}

// cuda_free_entry_impl: for free functions that pass a void* (host pointer directly).
// Looks up the alloc record to capture the size hint, but does NOT emit yet —
// the uretprobe checks rc and emits a free event only on CUDA_SUCCESS.
static __always_inline int cuda_free_entry_impl(void *ctx, void *ptr, u8 mem_kind, u64 stream_handle) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    u64 ptr_val = (u64)ptr;
    cuda_alloc_key_t key = {
        .ptr  = ptr_val,
        .tgid = (u32)(id >> 32),
        ._pad = 0,
    };
    cuda_alloc_record_t *rec = bpf_map_lookup_elem(&gpu_alloc_sizes, &key);
    cuda_ongoing_free_t inflight = {
        .ptr           = ptr_val,
        .size_hint     = rec ? rec->size : 0,
        .stream_handle = stream_handle,
        .mem_kind      = rec ? rec->mem_kind : mem_kind,
    };
    bpf_map_update_elem(&gpu_ongoing_free, &id, &inflight, BPF_ANY);
    return 0;
}

// cuda_free_entry_impl_u64: for cuMemFree_v2 / cuMemFreeAsync (CUdeviceptr arg)
static __always_inline int cuda_free_entry_impl_u64(void *ctx, u64 ptr_val, u8 mem_kind, u64 stream_handle) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    cuda_alloc_key_t key = {
        .ptr  = ptr_val,
        .tgid = (u32)(id >> 32),
        ._pad = 0,
    };
    cuda_alloc_record_t *rec = bpf_map_lookup_elem(&gpu_alloc_sizes, &key);
    cuda_ongoing_free_t inflight = {
        .ptr           = ptr_val,
        .size_hint     = rec ? rec->size : 0,
        .stream_handle = stream_handle,
        .mem_kind      = rec ? rec->mem_kind : mem_kind,
    };
    bpf_map_update_elem(&gpu_ongoing_free, &id, &inflight, BPF_ANY);
    return 0;
}

// cuda_free_exit_impl: uretprobe handler shared by all free variants.
// On CUDA_SUCCESS: emit cuda_free event and remove the pointer from
// gpu_alloc_sizes (so a failed free leaves the alloc record intact).
// On rc != 0: emit a gpu_cuda_errors_total event tagged with `func_id`.
static __always_inline int cuda_free_exit_impl(struct pt_regs *ctx, u8 func_id) {
    const u64 id = bpf_get_current_pid_tgid();
    cuda_ongoing_free_t *inflight = bpf_map_lookup_elem(&gpu_ongoing_free, &id);
    if (!inflight) {
        return 0;
    }
    s32 rc = (s32)(long)PT_REGS_RC(ctx);
    if (rc == 0) {
        cuda_free_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
        if (e) {
            e->flags         = k_event_free;
            e->mem_kind      = inflight->mem_kind;
            task_pid(&e->pid_info);
            e->size          = inflight->size_hint;
            e->stream_handle = inflight->stream_handle;
            bpf_ringbuf_submit(e, 0);
        }
        cuda_alloc_key_t key = {
            .ptr  = inflight->ptr,
            .tgid = (u32)(id >> 32),
            ._pad = 0,
        };
        bpf_map_delete_elem(&gpu_alloc_sizes, &key);
    } else {
        cuda_error_impl(ctx, func_id, rc);
    }
    bpf_map_delete_elem(&gpu_ongoing_free, &id);
    return 0;
}

// cu_memcpy_entry_impl: stash size+direction+stream; defer emit to uretprobe.
static __always_inline int cu_memcpy_entry_impl(void *ctx, u64 size, u8 direction, u64 stream_handle) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    cuda_ongoing_memcpy_t inflight = {
        .size          = (s64)size,
        .stream_handle = stream_handle,
        .direction     = direction,
    };
    bpf_map_update_elem(&gpu_ongoing_memcpy, &id, &inflight, BPF_ANY);
    return 0;
}

// cu_memcpy_exit_impl: emit memcpy event only when submission returned CUDA_SUCCESS.
// On rc != 0: emit a gpu_cuda_errors_total event tagged with `func_id`.
static __always_inline int cu_memcpy_exit_impl(struct pt_regs *ctx, u8 func_id) {
    const u64 id = bpf_get_current_pid_tgid();
    cuda_ongoing_memcpy_t *inflight = bpf_map_lookup_elem(&gpu_ongoing_memcpy, &id);
    if (!inflight) {
        return 0;
    }
    s32 rc = (s32)(long)PT_REGS_RC(ctx);
    if (rc == 0) {
        cuda_memcpy_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
        if (e) {
            e->flags         = k_event_memcpy;
            e->direction     = inflight->direction;
            task_pid(&e->pid_info);
            e->size          = inflight->size;
            e->stream_handle = inflight->stream_handle;
            bpf_ringbuf_submit(e, 0);
        }
    } else {
        cuda_error_impl(ctx, func_id, rc);
    }
    bpf_map_delete_elem(&gpu_ongoing_memcpy, &id);
    return 0;
}

// ────────────────────── Kernel launch probes ──────────────────────

// cu_launch_entry_impl: cuLaunchKernel passes 6 separate u32 dimension args
// (unlike cudaLaunchKernel which packed them into dim3 structs in registers)
static __always_inline int cu_launch_entry_impl(struct pt_regs *ctx, u64 func_handle,
                                                  u32 gX, u32 gY, u32 gZ,
                                                  u32 bX, u32 bY, u32 bZ,
                                                  u32 shared_mem_bytes, u64 stream_handle) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }

    u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&gpu_launch_start, &id, &ts, BPF_ANY);

    // HAMi throttle detection (only-HAMi mode): if libvgpu.so recorded a launch
    // entry timestamp, the delta is the time HAMi's rate_limiter() spent stalling.
    //
    // TTL guard: if libvgpu's wrapper rejected the launch (HAMi quota denied,
    // bad-arg early return) it never reaches libcuda, so the entry stays in
    // hami_launch_entry until the next launch on the same PID picks it up and
    // computes a bogus throttle duration. A single legitimate rate_limiter()
    // stall is bounded by HAMi's own retry policy (~hundreds of ms); 1 s is a
    // generous ceiling that filters out stale entries without affecting real
    // throttle measurements.
    #define HAMI_THROTTLE_MAX_NS (1ULL * 1000 * 1000 * 1000) // 1 second
    u64 *hami_t0 = bpf_map_lookup_elem(&hami_launch_entry, &id);
    if (hami_t0) {
        u64 stall = ts - *hami_t0;
        bpf_map_delete_elem(&hami_launch_entry, &id);
        if (stall <= HAMI_THROTTLE_MAX_NS) {
            hami_throttle_t *te = bpf_ringbuf_reserve(&gpu_events, sizeof(*te), 0);
            if (te) {
                te->flags       = k_event_hami_throttle;
                task_pid(&te->pid_info);
                te->duration_ns = stall;
                bpf_ringbuf_submit(te, 0);
            }
        }
        // else: stale entry from a rejected libvgpu launch — discard silently.
    }

    cuda_kernel_launch_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        return 0;
    }
    e->flags             = k_event_kernel_launch;
    task_pid(&e->pid_info);
    e->kern_func_off     = func_handle;
    e->grid_x            = (int)gX;
    e->grid_y            = (int)gY;
    e->grid_z            = (int)gZ;
    e->block_x           = (int)bX;
    e->block_y           = (int)bY;
    e->block_z           = (int)bZ;
    e->shared_mem_bytes  = shared_mem_bytes;
    e->_pad2             = 0;
    e->stream_handle     = stream_handle;
    bpf_ringbuf_submit(e, 0);
    return 0;
}

static __always_inline int cuda_launch_exit_impl(struct pt_regs *ctx, u8 func_id) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    u64 *entry_ts = bpf_map_lookup_elem(&gpu_launch_start, &id);
    if (!entry_ts) {
        return 0;
    }
    u64 now      = bpf_ktime_get_ns();
    s32 rc       = (s32)(long)PT_REGS_RC(ctx);
    u64 duration = now - *entry_ts;
    u64 saved_entry = *entry_ts;
    bpf_map_delete_elem(&gpu_launch_start, &id);

    cuda_kernel_launch_done_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (e) {
        e->flags       = k_event_kernel_launch_done;
        task_pid(&e->pid_info);
        e->entry_ts    = saved_entry;
        e->duration_ns = duration;
        e->retval      = rc;
        bpf_ringbuf_submit(e, 0);
    }
    if (rc != 0) {
        cuda_error_impl(ctx, func_id, rc);
    }
    return 0;
}

// cuLaunchKernel(CUfunction f, uint gX, uint gY, uint gZ,
//                uint bX, uint bY, uint bZ,                 // arg 7 (bZ) on stack
//                uint sharedMemBytes, CUstream hStream,     // args 8, 9 on stack
//                void **kernelParams, void **extra)         // args 10, 11 — not captured
// x86_64: first 6 args in RDI/RSI/RDX/RCX/R8/R9; stack args at SP+8, SP+16, SP+24, ...
// arm64:  first 8 args in x0..x7; remaining args on stack at SP+0, SP+8, ...
SEC("uprobe/cuLaunchKernel")
int BPF_KPROBE(obi_cu_launch,
               u64 fn,   // RDI: CUfunction handle
               u32 gX,   // RSI: gridDimX
               u32 gY,   // RDX: gridDimY
               u32 gZ,   // RCX: gridDimZ
               u32 bX,   // R8:  blockDimX
               u32 bY) { // R9:  blockDimY
    bpf_dbg_printk("=== uprobe/cuLaunchKernel id=%llx ===", bpf_get_current_pid_tgid());
    u32 bZ = 0;
    u32 sharedMem = 0;
    u64 hStream = 0;
#if defined(bpf_target_x86)
    // Stack args (after return-address pushed by `call`): arg7=SP+8, arg8=SP+16, arg9=SP+24
    bpf_probe_read_user(&bZ,        sizeof(bZ),        (void *)(PT_REGS_SP(ctx) + 8));
    bpf_probe_read_user(&sharedMem, sizeof(sharedMem), (void *)(PT_REGS_SP(ctx) + 16));
    bpf_probe_read_user(&hStream,   sizeof(hStream),   (void *)(PT_REGS_SP(ctx) + 24));
#elif defined(bpf_target_arm64)
    // arg7 (bZ) is x6, arg8 (sharedMem) is x7; arg9 (hStream) is first stack arg at SP+0
    bZ        = (u32)(((PT_REGS_ARM64 *)(ctx))->regs[6]);
    sharedMem = (u32)(((PT_REGS_ARM64 *)(ctx))->regs[7]);
    bpf_probe_read_user(&hStream, sizeof(hStream), (void *)(PT_REGS_SP(ctx) + 0));
#endif
    return cu_launch_entry_impl(ctx, fn, gX, gY, gZ, bX, bY, bZ, sharedMem, hStream);
}

SEC("uretprobe/cuLaunchKernel")
int BPF_KRETPROBE(obi_cu_launch_exit) {
    bpf_dbg_printk("=== uretprobe/cuLaunchKernel exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_launch_exit_impl(ctx, CUDA_FUNC_LAUNCH);
}

// cuLaunchCooperativeKernel has the same register layout as cuLaunchKernel
SEC("uprobe/cuLaunchCooperativeKernel")
int BPF_KPROBE(obi_cu_coop_launch,
               u64 fn,
               u32 gX,
               u32 gY,
               u32 gZ,
               u32 bX,
               u32 bY) {
    bpf_dbg_printk("=== uprobe/cuLaunchCooperativeKernel id=%llx ===", bpf_get_current_pid_tgid());
    u32 bZ = 0;
    u32 sharedMem = 0;
    u64 hStream = 0;
#if defined(bpf_target_x86)
    bpf_probe_read_user(&bZ,        sizeof(bZ),        (void *)(PT_REGS_SP(ctx) + 8));
    bpf_probe_read_user(&sharedMem, sizeof(sharedMem), (void *)(PT_REGS_SP(ctx) + 16));
    bpf_probe_read_user(&hStream,   sizeof(hStream),   (void *)(PT_REGS_SP(ctx) + 24));
#elif defined(bpf_target_arm64)
    bZ        = (u32)(((PT_REGS_ARM64 *)(ctx))->regs[6]);
    sharedMem = (u32)(((PT_REGS_ARM64 *)(ctx))->regs[7]);
    bpf_probe_read_user(&hStream, sizeof(hStream), (void *)(PT_REGS_SP(ctx) + 0));
#endif
    return cu_launch_entry_impl(ctx, fn, gX, gY, gZ, bX, bY, bZ, sharedMem, hStream);
}

SEC("uretprobe/cuLaunchCooperativeKernel")
int BPF_KRETPROBE(obi_cu_coop_launch_exit) {
    bpf_dbg_printk("=== uretprobe/cuLaunchCooperativeKernel exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_launch_exit_impl(ctx, CUDA_FUNC_COOP_LAUNCH);
}

// cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction fn,
//                  void **kernelParams, void **extra)
//
// CUlaunchConfig layout (CUDA 11.4+):
//   u32 gridDimX, gridDimY, gridDimZ;
//   u32 blockDimX, blockDimY, blockDimZ;
//   u32 sharedMemBytes;
//   CUstream hStream;                      // u64, 8-byte aligned → 4 bytes padding before
//   const CUlaunchAttribute *attrs;        // u64 (not captured)
//   u32 numAttrs;                          // u32 (not captured)
//
// We only read the first 9 fields (up to and including hStream).
// PyTorch 2.4+ Inductor backend and NCCL 2.20+ collectives launch through this API.
SEC("uprobe/cuLaunchKernelEx")
int BPF_KPROBE(obi_cu_launch_ex,
               void *config,   // RDI / x0: pointer to CUlaunchConfig
               u64 fn) {       // RSI / x1: CUfunction handle
    bpf_dbg_printk("=== uprobe/cuLaunchKernelEx id=%llx ===", bpf_get_current_pid_tgid());

    // Read CUlaunchConfig prefix (first 9 fields = 6 dims + sharedMem + pad + stream = 36 bytes).
    // We deliberately match the on-disk layout including the natural u32 → u64 padding.
    struct {
        u32 gX, gY, gZ;
        u32 bX, bY, bZ;
        u32 sharedMem;
        u32 _pad;          // CUlaunchConfig padding before u64 hStream
        u64 hStream;
    } cfg = {0};
    bpf_probe_read_user(&cfg, sizeof(cfg), config);

    return cu_launch_entry_impl(ctx, fn,
                                cfg.gX, cfg.gY, cfg.gZ,
                                cfg.bX, cfg.bY, cfg.bZ,
                                cfg.sharedMem, cfg.hStream);
}

SEC("uretprobe/cuLaunchKernelEx")
int BPF_KRETPROBE(obi_cu_launch_ex_exit) {
    bpf_dbg_printk("=== uretprobe/cuLaunchKernelEx exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_launch_exit_impl(ctx, CUDA_FUNC_LAUNCH);
}

// ──────────────────── Memory allocation probes ────────────────────

// cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize)
SEC("uprobe/cuMemAlloc_v2")
int BPF_KPROBE(obi_cu_mem_alloc, void **dptr, size_t size) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemAlloc_v2 id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_alloc_entry_impl(ctx, dptr, size, CUDA_MEM_KIND_DEVICE);
}

SEC("uretprobe/cuMemAlloc_v2")
int BPF_KRETPROBE(obi_cu_mem_alloc_exit) {
    return cuda_alloc_exit_impl(ctx, CUDA_FUNC_MALLOC);
}

// cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags)
SEC("uprobe/cuMemAllocManaged")
int BPF_KPROBE(obi_cu_mem_alloc_managed, void **dptr, size_t size) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemAllocManaged id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_alloc_entry_impl(ctx, dptr, size, CUDA_MEM_KIND_MANAGED);
}

SEC("uretprobe/cuMemAllocManaged")
int BPF_KRETPROBE(obi_cu_mem_alloc_managed_exit) {
    return cuda_alloc_exit_impl(ctx, CUDA_FUNC_MANAGED_MALLOC);
}

// cuMemAllocHost_v2(void** pp, size_t bytesize)
SEC("uprobe/cuMemAllocHost_v2")
int BPF_KPROBE(obi_cu_mem_alloc_host, void **pp, size_t size) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemAllocHost_v2 id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_alloc_entry_impl(ctx, pp, size, CUDA_MEM_KIND_HOST);
}

SEC("uretprobe/cuMemAllocHost_v2")
int BPF_KRETPROBE(obi_cu_mem_alloc_host_exit) {
    return cuda_alloc_exit_impl(ctx, CUDA_FUNC_HOST_MALLOC);
}

// cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags)
SEC("uprobe/cuMemHostAlloc")
int BPF_KPROBE(obi_cu_mem_host_alloc, void **pp, size_t size) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemHostAlloc id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_alloc_entry_impl(ctx, pp, size, CUDA_MEM_KIND_HOST);
}

SEC("uretprobe/cuMemHostAlloc")
int BPF_KRETPROBE(obi_cu_mem_host_alloc_exit) {
    return cuda_alloc_exit_impl(ctx, CUDA_FUNC_HOST_ALLOC);
}

// cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream)
SEC("uprobe/cuMemAllocAsync")
int BPF_KPROBE(obi_cu_mem_alloc_async, void **dptr, size_t size) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemAllocAsync id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_alloc_entry_impl(ctx, dptr, size, CUDA_MEM_KIND_POOL);
}

SEC("uretprobe/cuMemAllocAsync")
int BPF_KRETPROBE(obi_cu_mem_alloc_async_exit) {
    return cuda_alloc_exit_impl(ctx, CUDA_FUNC_ASYNC_MALLOC);
}

// cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream)
// Same as cuMemAllocAsync but with an explicit pool handle (arg 3) — used by PyTorch 2.1+ with
// PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync + custom MemoryPool, and by NCCL 2.19+ scratch
// buffers. Probing this is critical: without it, gpu_cuda_memory_allocations_bytes_total under-
// counts modern PyTorch workloads by 30–80%. Pool handle and stream arguments are ignored.
SEC("uprobe/cuMemAllocFromPoolAsync")
int BPF_KPROBE(obi_cu_mem_alloc_from_pool, void **dptr, size_t size) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemAllocFromPoolAsync id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_alloc_entry_impl(ctx, dptr, size, CUDA_MEM_KIND_POOL);
}

SEC("uretprobe/cuMemAllocFromPoolAsync")
int BPF_KRETPROBE(obi_cu_mem_alloc_from_pool_exit) {
    return cuda_alloc_exit_impl(ctx, CUDA_FUNC_POOL_MALLOC);
}

// ───────────────────────── Free probes ────────────────────────────
// All free probes use entry+exit pairs: entry stashes the inflight state,
// exit emits the event only when rc == CUDA_SUCCESS.

// cuMemFree_v2(CUdeviceptr dptr) — dptr is a u64 value, NOT a pointer
SEC("uprobe/cuMemFree_v2")
int BPF_KPROBE(obi_cu_mem_free, u64 dptr) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemFree_v2 id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_free_entry_impl_u64(ctx, dptr, CUDA_MEM_KIND_DEVICE, 0);
}
SEC("uretprobe/cuMemFree_v2")
int BPF_KRETPROBE(obi_cu_mem_free_exit) {
    bpf_dbg_printk("=== uretprobe/cuMemFree_v2 exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_free_exit_impl(ctx, CUDA_FUNC_FREE);
}

// cuMemFreeHost(void* p) — host pointer passed directly
SEC("uprobe/cuMemFreeHost")
int BPF_KPROBE(obi_cu_mem_free_host, void *p) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemFreeHost id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_free_entry_impl(ctx, p, CUDA_MEM_KIND_HOST, 0);
}
SEC("uretprobe/cuMemFreeHost")
int BPF_KRETPROBE(obi_cu_mem_free_host_exit) {
    bpf_dbg_printk("=== uretprobe/cuMemFreeHost exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_free_exit_impl(ctx, CUDA_FUNC_FREE_HOST);
}

// cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream)
SEC("uprobe/cuMemFreeAsync")
int BPF_KPROBE(obi_cu_mem_free_async, u64 dptr, u64 hStream) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemFreeAsync id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_free_entry_impl_u64(ctx, dptr, CUDA_MEM_KIND_POOL, hStream);
}
SEC("uretprobe/cuMemFreeAsync")
int BPF_KRETPROBE(obi_cu_mem_free_async_exit) {
    bpf_dbg_printk("=== uretprobe/cuMemFreeAsync exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_free_exit_impl(ctx, CUDA_FUNC_FREE_ASYNC);
}

// cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags)
// Pins existing host memory so the GPU can DMA directly to/from it.
// Used heavily by NCCL for inter-GPU communication staging buffers and by
// PyTorch DataLoader workers with pin_memory=True. Counted as a HOST alloc
// for metric purposes; cuMemHostUnregister pairs with cuMemFreeHost-style free.
SEC("uprobe/cuMemHostRegister")
int BPF_KPROBE(obi_cu_mem_host_register, void *p, size_t size) {
    (void)ctx;
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) { return 0; }
    bpf_dbg_printk("=== uprobe/cuMemHostRegister id=%llx ===", id);
    cuda_ongoing_host_reg_t inflight = { .ptr = (u64)p, .size = (s64)size };
    bpf_map_update_elem(&gpu_ongoing_host_reg, &id, &inflight, BPF_ANY);
    return 0;
}

SEC("uretprobe/cuMemHostRegister")
int BPF_KRETPROBE(obi_cu_mem_host_register_exit) {
    const u64 id = bpf_get_current_pid_tgid();
    cuda_ongoing_host_reg_t *inflight = bpf_map_lookup_elem(&gpu_ongoing_host_reg, &id);
    if (!inflight) { return 0; }
    s32 rc = (s32)(long)PT_REGS_RC(ctx);
    if (rc == 0) {
        // Emit malloc event with HOST kind
        cuda_malloc_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
        if (e) {
            e->flags    = k_event_malloc;
            e->mem_kind = CUDA_MEM_KIND_HOST;
            task_pid(&e->pid_info);
            e->size     = inflight->size;
            bpf_ringbuf_submit(e, 0);
        }
        // Record in gpu_alloc_sizes so cuMemHostUnregister can later report bytes
        cuda_alloc_key_t key = {
            .ptr  = inflight->ptr,
            .tgid = (u32)(id >> 32),
            ._pad = 0,
        };
        cuda_alloc_record_t alloc_rec = {
            .size     = inflight->size,
            .mem_kind = CUDA_MEM_KIND_HOST,
        };
        bpf_map_update_elem(&gpu_alloc_sizes, &key, &alloc_rec, BPF_ANY);
    } else {
        cuda_error_impl(ctx, CUDA_FUNC_HOST_REGISTER, rc);
    }
    bpf_map_delete_elem(&gpu_ongoing_host_reg, &id);
    return 0;
}

// cuMemHostUnregister(void *p) — un-pin previously registered host memory.
// Reuses the standard free entry+exit pair (which looks up the pointer in
// gpu_alloc_sizes and emits a free event with the recorded kind/size).
SEC("uprobe/cuMemHostUnregister")
int BPF_KPROBE(obi_cu_mem_host_unregister, void *p) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemHostUnregister id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_free_entry_impl(ctx, p, CUDA_MEM_KIND_HOST, 0);
}
SEC("uretprobe/cuMemHostUnregister")
int BPF_KRETPROBE(obi_cu_mem_host_unregister_exit) {
    return cuda_free_exit_impl(ctx, CUDA_FUNC_HOST_UNREGISTER);
}

// ──────────────────────── Memcpy probes ───────────────────────────
// All async memcpy variants use entry+exit; entry stashes (size, direction, stream),
// exit emits only when rc == CUDA_SUCCESS (i.e. submission accepted).

// cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream)
SEC("uprobe/cuMemcpyHtoDAsync_v2")
int BPF_KPROBE(obi_cu_memcpy_htod, u64 dst, void *src, size_t size, u64 hStream) {
    (void)ctx;
    (void)dst;
    (void)src;
    bpf_dbg_printk("=== uprobe/cuMemcpyHtoDAsync_v2 id=%llx size=%lu ===", bpf_get_current_pid_tgid(), size);
    return cu_memcpy_entry_impl(ctx, (u64)size, CUDA_MEMCPY_DIR_HTOD, hStream);
}
SEC("uretprobe/cuMemcpyHtoDAsync_v2")
int BPF_KRETPROBE(obi_cu_memcpy_htod_exit) {
    return cu_memcpy_exit_impl(ctx, CUDA_FUNC_MEMCPY_HTOD);
}

// cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
SEC("uprobe/cuMemcpyDtoHAsync_v2")
int BPF_KPROBE(obi_cu_memcpy_dtoh, void *dst, u64 src, size_t size, u64 hStream) {
    (void)ctx;
    (void)dst;
    (void)src;
    bpf_dbg_printk("=== uprobe/cuMemcpyDtoHAsync_v2 id=%llx size=%lu ===", bpf_get_current_pid_tgid(), size);
    return cu_memcpy_entry_impl(ctx, (u64)size, CUDA_MEMCPY_DIR_DTOH, hStream);
}
SEC("uretprobe/cuMemcpyDtoHAsync_v2")
int BPF_KRETPROBE(obi_cu_memcpy_dtoh_exit) {
    return cu_memcpy_exit_impl(ctx, CUDA_FUNC_MEMCPY_DTOH);
}

// cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
SEC("uprobe/cuMemcpyDtoDAsync_v2")
int BPF_KPROBE(obi_cu_memcpy_dtod, u64 dst, u64 src, size_t size, u64 hStream) {
    (void)ctx;
    (void)dst;
    (void)src;
    bpf_dbg_printk("=== uprobe/cuMemcpyDtoDAsync_v2 id=%llx size=%lu ===", bpf_get_current_pid_tgid(), size);
    return cu_memcpy_entry_impl(ctx, (u64)size, CUDA_MEMCPY_DIR_DTOD, hStream);
}
SEC("uretprobe/cuMemcpyDtoDAsync_v2")
int BPF_KRETPROBE(obi_cu_memcpy_dtod_exit) {
    return cu_memcpy_exit_impl(ctx, CUDA_FUNC_MEMCPY_DTOD);
}

// cuMemcpyPeer(CUdeviceptr dst, CUcontext dstCtx, CUdeviceptr src, CUcontext srcCtx, size_t count)
// CUcontext args are opaque handles (not device ordinals); src/dst device fields stay 0
// until a future plan resolves CUcontext → device ordinal via cuCtxGetDevice probing.
// Entry stashes size+stream; exit emits only on rc == CUDA_SUCCESS.
SEC("uprobe/cuMemcpyPeer")
int BPF_KPROBE(obi_cu_memcpy_peer, u64 dst, u64 dstCtx, u64 src, u64 srcCtx, size_t size) {
    (void)ctx;
    (void)dst;
    (void)dstCtx;
    (void)src;
    (void)srcCtx;

    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    bpf_dbg_printk("=== uprobe/cuMemcpyPeer id=%llx size=%lu ===", id, size);

    cuda_ongoing_peer_copy_t inflight = {
        .size          = (s64)size,
        .stream_handle = 0,
        .src_device    = 0,
        .dst_device    = 0,
    };
    bpf_map_update_elem(&gpu_ongoing_peer_copy, &id, &inflight, BPF_ANY);
    return 0;
}

// cuMemcpyPeerAsync(... , CUstream hStream) — 6th arg is the stream handle (stack on x86_64)
SEC("uprobe/cuMemcpyPeerAsync")
int BPF_KPROBE(obi_cu_memcpy_peer_async, u64 dst, u64 dstCtx, u64 src, u64 srcCtx, size_t size) {
    (void)ctx;
    (void)dst;
    (void)dstCtx;
    (void)src;
    (void)srcCtx;

    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    bpf_dbg_printk("=== uprobe/cuMemcpyPeerAsync id=%llx size=%lu ===", id, size);

    u64 hStream = 0;
#if defined(bpf_target_x86)
    // Args 1..5 in regs, arg 6 (hStream) is the first stack arg at SP+8
    bpf_probe_read_user(&hStream, sizeof(hStream), (void *)(PT_REGS_SP(ctx) + 8));
#elif defined(bpf_target_arm64)
    // Args 1..5 in x0..x4, arg 6 in x5
    hStream = (u64)(((PT_REGS_ARM64 *)(ctx))->regs[5]);
#endif

    cuda_ongoing_peer_copy_t inflight = {
        .size          = (s64)size,
        .stream_handle = hStream,
        .src_device    = 0,
        .dst_device    = 0,
    };
    bpf_map_update_elem(&gpu_ongoing_peer_copy, &id, &inflight, BPF_ANY);
    return 0;
}

// Shared peer copy exit handler
// On rc != 0: emit a gpu_cuda_errors_total event tagged with `func_id`.
static __always_inline int cu_memcpy_peer_exit_impl(struct pt_regs *ctx, u8 func_id) {
    const u64 id = bpf_get_current_pid_tgid();
    cuda_ongoing_peer_copy_t *inflight = bpf_map_lookup_elem(&gpu_ongoing_peer_copy, &id);
    if (!inflight) {
        return 0;
    }
    s32 rc = (s32)(long)PT_REGS_RC(ctx);
    if (rc == 0) {
        cuda_peer_copy_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
        if (e) {
            e->flags         = k_event_peer_copy;
            e->src_device    = inflight->src_device;
            e->dst_device    = inflight->dst_device;
            task_pid(&e->pid_info);
            e->size          = inflight->size;
            e->stream_handle = inflight->stream_handle;
            bpf_ringbuf_submit(e, 0);
        }
    } else {
        cuda_error_impl(ctx, func_id, rc);
    }
    bpf_map_delete_elem(&gpu_ongoing_peer_copy, &id);
    return 0;
}

SEC("uretprobe/cuMemcpyPeer")
int BPF_KRETPROBE(obi_cu_memcpy_peer_exit) {
    return cu_memcpy_peer_exit_impl(ctx, CUDA_FUNC_MEMCPY_PEER);
}
SEC("uretprobe/cuMemcpyPeerAsync")
int BPF_KRETPROBE(obi_cu_memcpy_peer_async_exit) {
    return cu_memcpy_peer_exit_impl(ctx, CUDA_FUNC_MEMCPY_PEER_ASYNC);
}

// ───────────────────────── Memset probes ──────────────────────────
// Entry stashes (size, is_async, stream); exit emits only on rc == CUDA_SUCCESS.

static __always_inline int cu_memset_entry_impl(void *ctx, size_t count, u8 is_async, u64 stream_handle) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    cuda_ongoing_memset_t inflight = {
        .size          = (s64)count,
        .stream_handle = stream_handle,
        .is_async      = is_async,
    };
    bpf_map_update_elem(&gpu_ongoing_memset, &id, &inflight, BPF_ANY);
    return 0;
}

// On rc != 0: emit a gpu_cuda_errors_total event tagged with `func_id`.
static __always_inline int cu_memset_exit_impl(struct pt_regs *ctx, u8 func_id) {
    const u64 id = bpf_get_current_pid_tgid();
    cuda_ongoing_memset_t *inflight = bpf_map_lookup_elem(&gpu_ongoing_memset, &id);
    if (!inflight) {
        return 0;
    }
    s32 rc = (s32)(long)PT_REGS_RC(ctx);
    if (rc == 0) {
        cuda_memset_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
        if (e) {
            e->flags         = k_event_memset;
            e->is_async      = inflight->is_async;
            task_pid(&e->pid_info);
            e->size          = inflight->size;
            e->stream_handle = inflight->stream_handle;
            bpf_ringbuf_submit(e, 0);
        }
    } else {
        cuda_error_impl(ctx, func_id, rc);
    }
    bpf_map_delete_elem(&gpu_ongoing_memset, &id);
    return 0;
}

// cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N)
SEC("uprobe/cuMemsetD8_v2")
int BPF_KPROBE(obi_cu_memset, u64 devPtr, u8 value, size_t count) {
    (void)ctx;
    (void)devPtr;
    (void)value;
    bpf_dbg_printk("=== uprobe/cuMemsetD8_v2 id=%llx count=%lu ===", bpf_get_current_pid_tgid(), count);
    return cu_memset_entry_impl(ctx, count, 0 /*sync*/, 0);
}
SEC("uretprobe/cuMemsetD8_v2")
int BPF_KRETPROBE(obi_cu_memset_exit) {
    return cu_memset_exit_impl(ctx, CUDA_FUNC_MEMSET);
}

// cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
SEC("uprobe/cuMemsetD8Async")
int BPF_KPROBE(obi_cu_memset_async, u64 devPtr, u8 value, size_t count, u64 hStream) {
    (void)ctx;
    (void)devPtr;
    (void)value;
    bpf_dbg_printk("=== uprobe/cuMemsetD8Async id=%llx count=%lu ===", bpf_get_current_pid_tgid(), count);
    return cu_memset_entry_impl(ctx, count, 1 /*async*/, hStream);
}
SEC("uretprobe/cuMemsetD8Async")
int BPF_KRETPROBE(obi_cu_memset_async_exit) {
    return cu_memset_exit_impl(ctx, CUDA_FUNC_MEMSET_ASYNC);
}

// ─────────────────────── Graph launch probe ───────────────────────
// The entry uprobe emits a graph_launch event regardless of whether the
// submission succeeds (matches the cuda_kernel_launch_t entry pattern). The
// uretprobe is error-only: on rc != 0 it emits a cuda_error_t so failed graph
// launches show up in gpu_cuda_errors_total. No inflight map is needed since
// the uretprobe carries the return code and nothing else.

SEC("uprobe/cuGraphLaunch")
int BPF_KPROBE(obi_cu_graph_launch) {
    (void)ctx;
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }

    bpf_dbg_printk("=== uprobe/cuGraphLaunch id=%llx ===", id);

    cuda_graph_launch_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        return 0;
    }
    e->flags = k_event_graph_launch;
    task_pid(&e->pid_info);
    bpf_ringbuf_submit(e, 0);
    return 0;
}

SEC("uretprobe/cuGraphLaunch")
int BPF_KRETPROBE(obi_cu_graph_launch_exit) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    s32 rc = (s32)(long)PT_REGS_RC(ctx);
    if (rc != 0) {
        cuda_error_impl(ctx, CUDA_FUNC_GRAPH_LAUNCH, rc);
    }
    return 0;
}

// ──────────────────── Synchronize probes (entry + exit) ────────────

// cuStreamSynchronize(CUstream hStream) — hStream in RDI / x0
SEC("uprobe/cuStreamSynchronize")
int BPF_KPROBE(obi_cu_stream_sync, u64 hStream) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuStreamSynchronize entry id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_sync_entry_impl(ctx, hStream);
}

SEC("uretprobe/cuStreamSynchronize")
int BPF_KRETPROBE(obi_cu_stream_sync_exit) {
    bpf_dbg_printk("=== uretprobe/cuStreamSynchronize exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_sync_exit_impl(ctx, CUDA_SYNC_KIND_STREAM, CUDA_FUNC_SYNC_STREAM);
}

// cuCtxSynchronize(void) — current context not exposed in registers; emit handle=0
// (proper context resolution requires probing cuCtxSetCurrent/Push to maintain a
// per-PID stack — deferred to a future plan).
SEC("uprobe/cuCtxSynchronize")
int BPF_KPROBE(obi_cu_ctx_sync) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuCtxSynchronize entry id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_sync_entry_impl(ctx, 0);
}

SEC("uretprobe/cuCtxSynchronize")
int BPF_KRETPROBE(obi_cu_ctx_sync_exit) {
    bpf_dbg_printk("=== uretprobe/cuCtxSynchronize exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_sync_exit_impl(ctx, CUDA_SYNC_KIND_DEVICE, CUDA_FUNC_SYNC_DEVICE);
}

// cuEventSynchronize(CUevent hEvent) — hEvent in RDI / x0
SEC("uprobe/cuEventSynchronize")
int BPF_KPROBE(obi_cu_event_sync, u64 hEvent) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuEventSynchronize entry id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_sync_entry_impl(ctx, hEvent);
}

SEC("uretprobe/cuEventSynchronize")
int BPF_KRETPROBE(obi_cu_event_sync_exit) {
    bpf_dbg_printk("=== uretprobe/cuEventSynchronize exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_sync_exit_impl(ctx, CUDA_SYNC_KIND_EVENT, CUDA_FUNC_SYNC_EVENT);
}

// ─────────────── cuEventElapsedTime probe (GPU time capture) ───────────────
// cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd)
// arg 1 (RDI / x0): pointer to a float that receives the elapsed milliseconds
// arg 2 (RSI / x1): start event handle
// arg 3 (RDX / x2): end event handle
//
// The probe captures the application's own measurement of GPU execution time
// between two events. This is the ONLY way to measure on-device runtime without
// CUPTI — PyTorch profiler / FlashAttention benchmarks / vLLM all use this idiom.
SEC("uprobe/cuEventElapsedTime")
int BPF_KPROBE(obi_cu_event_elapsed, void *p_ms, u64 h_start, u64 h_end) {
    (void)ctx;
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) { return 0; }
    bpf_dbg_printk("=== uprobe/cuEventElapsedTime id=%llx ===", id);
    cuda_ongoing_event_elapsed_t inflight = {
        .p_milliseconds_addr = (u64)p_ms,
        .h_start             = h_start,
        .h_end               = h_end,
    };
    bpf_map_update_elem(&gpu_ongoing_event_elapsed, &id, &inflight, BPF_ANY);
    return 0;
}

SEC("uretprobe/cuEventElapsedTime")
int BPF_KRETPROBE(obi_cu_event_elapsed_exit) {
    const u64 id = bpf_get_current_pid_tgid();
    cuda_ongoing_event_elapsed_t *inf = bpf_map_lookup_elem(&gpu_ongoing_event_elapsed, &id);
    if (!inf) { return 0; }
    s32 rc = (s32)(long)PT_REGS_RC(ctx);
    if (rc == 0) {
        u32 ms_bits = 0;
        if (bpf_probe_read_user(&ms_bits, sizeof(ms_bits), (void *)inf->p_milliseconds_addr) == 0) {
            cuda_event_elapsed_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
            if (e) {
                e->flags           = k_event_event_elapsed;
                task_pid(&e->pid_info);
                e->elapsed_ms_bits = ms_bits;
                e->_pad2           = 0;
                e->h_start         = inf->h_start;
                e->h_end           = inf->h_end;
                bpf_ringbuf_submit(e, 0);
            }
        }
    } else {
        cuda_error_impl(ctx, CUDA_FUNC_EVENT_ELAPSED, rc);
    }
    bpf_map_delete_elem(&gpu_ongoing_event_elapsed, &id);
    return 0;
}
