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
};

// ──────────────────────────── BPF maps ────────────────────────────

// Maps pid_tgid → entry timestamp for in-flight sync calls
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key,   u64);
    __type(value, u64);
} gpu_sync_start SEC(".maps");

// Maps pid_tgid → {devptr_addr, size, mem_kind} for in-flight alloc calls
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key,   u64);
    __type(value, cuda_ongoing_alloc_t);
} gpu_ongoing_alloc SEC(".maps");

// Maps device pointer → {size, mem_kind} so cuMemFree can report bytes freed
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 131072);
    __type(key,   u64);
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

static __always_inline int cuda_sync_entry_impl(void *ctx) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&gpu_sync_start, &id, &ts, BPF_ANY);
    return 0;
}

static __always_inline int cuda_sync_exit_impl(struct pt_regs *ctx, u8 sync_kind, u8 func_id) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    u64 *entry_ts = bpf_map_lookup_elem(&gpu_sync_start, &id);
    if (!entry_ts) {
        return 0;
    }
    u64 now = bpf_ktime_get_ns();
    s32 rc = (s32)(long)PT_REGS_RC(ctx);
    u64 duration = now - *entry_ts;
    u64 saved_entry = *entry_ts;
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
            bpf_map_update_elem(&gpu_alloc_sizes, &ptr_val, &alloc_rec, BPF_ANY);
        }
    } else {
        cuda_error_impl(ctx, func_id, rc);
    }
    bpf_map_delete_elem(&gpu_ongoing_alloc, &id);
    return 0;
}

// cuda_free_impl: for free functions that pass a void* (host pointer directly)
static __always_inline int cuda_free_impl(void *ctx, void *ptr, u8 mem_kind) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    u64 ptr_val = (u64)ptr;
    cuda_alloc_record_t *rec = bpf_map_lookup_elem(&gpu_alloc_sizes, &ptr_val);

    cuda_free_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        if (rec) {
            bpf_map_delete_elem(&gpu_alloc_sizes, &ptr_val);
        }
        return 0;
    }
    e->flags    = k_event_free;
    e->mem_kind = mem_kind;
    task_pid(&e->pid_info);
    e->size = rec ? rec->size : 0;
    bpf_ringbuf_submit(e, 0);

    if (rec) {
        bpf_map_delete_elem(&gpu_alloc_sizes, &ptr_val);
    }
    return 0;
}

// cuda_free_impl_u64: for cuMemFree_v2 / cuMemFreeAsync where arg is CUdeviceptr (u64 value)
static __always_inline int cuda_free_impl_u64(void *ctx, u64 ptr_val, u8 mem_kind) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    cuda_alloc_record_t *rec = bpf_map_lookup_elem(&gpu_alloc_sizes, &ptr_val);

    cuda_free_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        if (rec) {
            bpf_map_delete_elem(&gpu_alloc_sizes, &ptr_val);
        }
        return 0;
    }
    e->flags    = k_event_free;
    e->mem_kind = mem_kind;
    task_pid(&e->pid_info);
    e->size = rec ? rec->size : 0;
    bpf_ringbuf_submit(e, 0);

    if (rec) {
        bpf_map_delete_elem(&gpu_alloc_sizes, &ptr_val);
    }
    return 0;
}

// cu_memcpy_impl: emits a memcpy event with the given direction constant
static __always_inline int cu_memcpy_impl(void *ctx, u64 size, u8 direction) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    cuda_memcpy_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        return 0;
    }
    e->flags     = k_event_memcpy;
    e->direction = direction;
    task_pid(&e->pid_info);
    e->size      = (s64)size;
    bpf_ringbuf_submit(e, 0);
    return 0;
}

// ────────────────────── Kernel launch probes ──────────────────────

// cu_launch_entry_impl: cuLaunchKernel passes 6 separate u32 dimension args
// (unlike cudaLaunchKernel which packed them into dim3 structs in registers)
static __always_inline int cu_launch_entry_impl(struct pt_regs *ctx, u64 func_handle,
                                                  u32 gX, u32 gY, u32 gZ,
                                                  u32 bX, u32 bY, u32 bZ) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }

    u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&gpu_launch_start, &id, &ts, BPF_ANY);

    // HAMi throttle detection (only-HAMi mode): if libvgpu.so recorded a launch
    // entry timestamp, the delta is the time HAMi's rate_limiter() spent stalling.
    u64 *hami_t0 = bpf_map_lookup_elem(&hami_launch_entry, &id);
    if (hami_t0) {
        hami_throttle_t *te = bpf_ringbuf_reserve(&gpu_events, sizeof(*te), 0);
        if (te) {
            te->flags       = k_event_hami_throttle;
            task_pid(&te->pid_info);
            te->duration_ns = ts - *hami_t0;
            bpf_ringbuf_submit(te, 0);
        }
        bpf_map_delete_elem(&hami_launch_entry, &id);
    }

    cuda_kernel_launch_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        return 0;
    }
    e->flags         = k_event_kernel_launch;
    task_pid(&e->pid_info);
    e->kern_func_off = func_handle;
    e->grid_x        = (int)gX;
    e->grid_y        = (int)gY;
    e->grid_z        = (int)gZ;
    e->block_x       = (int)bX;
    e->block_y       = (int)bY;
    e->block_z       = (int)bZ;
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

// cuLaunchKernel(CUfunction f, uint gX, uint gY, uint gZ, uint bX, uint bY, uint bZ, ...)
// x86_64: first 6 args in RDI/RSI/RDX/RCX/R8/R9; blockDimZ (arg7) is on stack at SP+8
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
#if defined(bpf_target_x86)
    // blockDimZ is the 7th argument; after the call instruction RSP points to
    // the return address, so the first stack arg is at RSP+8.
    bpf_probe_read_user(&bZ, sizeof(bZ), (void *)(PT_REGS_SP(ctx) + 8));
#elif defined(bpf_target_arm64)
    // arm64: first 8 args in x0-x7; blockDimZ is in x6 (regs[6])
    bZ = (u32)(((PT_REGS_ARM64 *)(ctx))->regs[6]);
#endif
    return cu_launch_entry_impl(ctx, fn, gX, gY, gZ, bX, bY, bZ);
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
#if defined(bpf_target_x86)
    bpf_probe_read_user(&bZ, sizeof(bZ), (void *)(PT_REGS_SP(ctx) + 8));
#elif defined(bpf_target_arm64)
    bZ = (u32)(((PT_REGS_ARM64 *)(ctx))->regs[6]);
#endif
    return cu_launch_entry_impl(ctx, fn, gX, gY, gZ, bX, bY, bZ);
}

SEC("uretprobe/cuLaunchCooperativeKernel")
int BPF_KRETPROBE(obi_cu_coop_launch_exit) {
    bpf_dbg_printk("=== uretprobe/cuLaunchCooperativeKernel exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_launch_exit_impl(ctx, CUDA_FUNC_COOP_LAUNCH);
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

// ───────────────────────── Free probes ────────────────────────────

// cuMemFree_v2(CUdeviceptr dptr) — dptr is a u64 value, NOT a pointer
SEC("uprobe/cuMemFree_v2")
int BPF_KPROBE(obi_cu_mem_free, u64 dptr) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemFree_v2 id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_free_impl_u64(ctx, dptr, CUDA_MEM_KIND_DEVICE);
}

// cuMemFreeHost(void* p) — host pointer passed directly
SEC("uprobe/cuMemFreeHost")
int BPF_KPROBE(obi_cu_mem_free_host, void *p) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemFreeHost id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_free_impl(ctx, p, CUDA_MEM_KIND_HOST);
}

// cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream)
SEC("uprobe/cuMemFreeAsync")
int BPF_KPROBE(obi_cu_mem_free_async, u64 dptr) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuMemFreeAsync id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_free_impl_u64(ctx, dptr, CUDA_MEM_KIND_POOL);
}

// ──────────────────────── Memcpy probes ───────────────────────────

// cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream)
SEC("uprobe/cuMemcpyHtoDAsync_v2")
int BPF_KPROBE(obi_cu_memcpy_htod, u64 dst, void *src, size_t size) {
    (void)ctx;
    (void)dst;
    (void)src;
    bpf_dbg_printk("=== uprobe/cuMemcpyHtoDAsync_v2 id=%llx size=%lu ===", bpf_get_current_pid_tgid(), size);
    return cu_memcpy_impl(ctx, (u64)size, CUDA_MEMCPY_DIR_HTOD);
}

// cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream)
SEC("uprobe/cuMemcpyDtoHAsync_v2")
int BPF_KPROBE(obi_cu_memcpy_dtoh, void *dst, u64 src, size_t size) {
    (void)ctx;
    (void)dst;
    (void)src;
    bpf_dbg_printk("=== uprobe/cuMemcpyDtoHAsync_v2 id=%llx size=%lu ===", bpf_get_current_pid_tgid(), size);
    return cu_memcpy_impl(ctx, (u64)size, CUDA_MEMCPY_DIR_DTOH);
}

// cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream)
SEC("uprobe/cuMemcpyDtoDAsync_v2")
int BPF_KPROBE(obi_cu_memcpy_dtod, u64 dst, u64 src, size_t size) {
    (void)ctx;
    (void)dst;
    (void)src;
    bpf_dbg_printk("=== uprobe/cuMemcpyDtoDAsync_v2 id=%llx size=%lu ===", bpf_get_current_pid_tgid(), size);
    return cu_memcpy_impl(ctx, (u64)size, CUDA_MEMCPY_DIR_DTOD);
}

// cuMemcpyPeer(CUdeviceptr dst, CUcontext dstCtx, CUdeviceptr src, CUcontext srcCtx, size_t count)
// CUcontext args are opaque handles (not device ordinals); zero src/dst device fields.
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

    cuda_peer_copy_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        return 0;
    }
    e->flags      = k_event_peer_copy;
    e->src_device = 0; // CUcontext is opaque; device ordinal not available
    e->dst_device = 0;
    task_pid(&e->pid_info);
    e->size = (s64)size;
    bpf_ringbuf_submit(e, 0);
    return 0;
}

// cuMemcpyPeerAsync has the same first 5 args as cuMemcpyPeer; stream is arg 6 (ignored)
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

    cuda_peer_copy_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        return 0;
    }
    e->flags      = k_event_peer_copy;
    e->src_device = 0;
    e->dst_device = 0;
    task_pid(&e->pid_info);
    e->size = (s64)size;
    bpf_ringbuf_submit(e, 0);
    return 0;
}

// ───────────────────────── Memset probes ──────────────────────────

// cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N)
SEC("uprobe/cuMemsetD8_v2")
int BPF_KPROBE(obi_cu_memset, u64 devPtr, u8 value, size_t count) {
    (void)ctx;
    (void)devPtr;
    (void)value;

    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }

    bpf_dbg_printk("=== uprobe/cuMemsetD8_v2 id=%llx count=%lu ===", id, count);

    cuda_memset_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        return 0;
    }
    e->flags    = k_event_memset;
    e->is_async = 0;
    task_pid(&e->pid_info);
    e->size = (s64)count;
    bpf_ringbuf_submit(e, 0);
    return 0;
}

// cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
SEC("uprobe/cuMemsetD8Async")
int BPF_KPROBE(obi_cu_memset_async, u64 devPtr, u8 value, size_t count) {
    (void)ctx;
    (void)devPtr;
    (void)value;

    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }

    bpf_dbg_printk("=== uprobe/cuMemsetD8Async id=%llx count=%lu ===", id, count);

    cuda_memset_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        return 0;
    }
    e->flags    = k_event_memset;
    e->is_async = 1;
    task_pid(&e->pid_info);
    e->size = (s64)count;
    bpf_ringbuf_submit(e, 0);
    return 0;
}

// ─────────────────────── Graph launch probe ───────────────────────

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

// ──────────────────── Synchronize probes (entry + exit) ────────────

SEC("uprobe/cuStreamSynchronize")
int BPF_KPROBE(obi_cu_stream_sync) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuStreamSynchronize entry id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_sync_entry_impl(ctx);
}

SEC("uretprobe/cuStreamSynchronize")
int BPF_KRETPROBE(obi_cu_stream_sync_exit) {
    bpf_dbg_printk("=== uretprobe/cuStreamSynchronize exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_sync_exit_impl(ctx, CUDA_SYNC_KIND_STREAM, CUDA_FUNC_SYNC_STREAM);
}

// cuCtxSynchronize() is the Driver API equivalent of cudaDeviceSynchronize()
SEC("uprobe/cuCtxSynchronize")
int BPF_KPROBE(obi_cu_ctx_sync) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuCtxSynchronize entry id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_sync_entry_impl(ctx);
}

SEC("uretprobe/cuCtxSynchronize")
int BPF_KRETPROBE(obi_cu_ctx_sync_exit) {
    bpf_dbg_printk("=== uretprobe/cuCtxSynchronize exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_sync_exit_impl(ctx, CUDA_SYNC_KIND_DEVICE, CUDA_FUNC_SYNC_DEVICE);
}

SEC("uprobe/cuEventSynchronize")
int BPF_KPROBE(obi_cu_event_sync) {
    (void)ctx;
    bpf_dbg_printk("=== uprobe/cuEventSynchronize entry id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_sync_entry_impl(ctx);
}

SEC("uretprobe/cuEventSynchronize")
int BPF_KRETPROBE(obi_cu_event_sync_exit) {
    bpf_dbg_printk("=== uretprobe/cuEventSynchronize exit id=%llx ===", bpf_get_current_pid_tgid());
    return cuda_sync_exit_impl(ctx, CUDA_SYNC_KIND_EVENT, CUDA_FUNC_SYNC_EVENT);
}
