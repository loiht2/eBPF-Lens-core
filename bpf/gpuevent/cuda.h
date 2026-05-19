// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Source: https://github.com/facebookincubator/strobelight/blob/5d84bcfdd9abccc615b45a390bfd7bba7097dc51/strobelight/src/profilers/gpuevent_snoop/bpf/gpuevent_snoop.hO

// Copyright (c) Meta Platforms, Inc. and affiliates.
// Copyright Grafana Labs
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <pid/types/pid_info.h>

// Memory kind values for cuda_malloc_t and cuda_free_t
#define CUDA_MEM_KIND_DEVICE  1
#define CUDA_MEM_KIND_HOST    2
#define CUDA_MEM_KIND_MANAGED 3
#define CUDA_MEM_KIND_POOL    4

// Memcpy direction values for cuda_memcpy_t.direction
#define CUDA_MEMCPY_DIR_HTOD 1
#define CUDA_MEMCPY_DIR_DTOH 2
#define CUDA_MEMCPY_DIR_DTOD 3

// Sync kind values for cuda_sync_t
#define CUDA_SYNC_KIND_STREAM 1
#define CUDA_SYNC_KIND_DEVICE 2
#define CUDA_SYNC_KIND_EVENT  3

// Function ID values for cuda_error_t
#define CUDA_FUNC_LAUNCH         1
#define CUDA_FUNC_COOP_LAUNCH    2
#define CUDA_FUNC_MALLOC         3
#define CUDA_FUNC_MANAGED_MALLOC 4
#define CUDA_FUNC_HOST_MALLOC    5
#define CUDA_FUNC_HOST_ALLOC     6
#define CUDA_FUNC_ASYNC_MALLOC   7
#define CUDA_FUNC_SYNC_STREAM    8
#define CUDA_FUNC_SYNC_DEVICE    9
#define CUDA_FUNC_SYNC_EVENT     10
#define CUDA_FUNC_POOL_MALLOC    11 // cuMemAllocFromPoolAsync (PyTorch 2.1+ custom pool)
#define CUDA_FUNC_HOST_REGISTER  12 // cuMemHostRegister (pin existing host memory; NCCL staging buffers)
#define CUDA_FUNC_HOST_UNREGISTER 13 // cuMemHostUnregister
#define CUDA_FUNC_EVENT_ELAPSED  14 // cuEventElapsedTime (GPU time measurement)
// v0.12: extended error coverage. The functions below previously only emitted
// success events; their uretprobes now also emit gpu_cuda_errors_total when
// rc != CUDA_SUCCESS, so the dashboard's CUDA Errors panel covers them too.
#define CUDA_FUNC_FREE              15 // cuMemFree_v2
#define CUDA_FUNC_FREE_HOST         16 // cuMemFreeHost
#define CUDA_FUNC_FREE_ASYNC        17 // cuMemFreeAsync
#define CUDA_FUNC_MEMCPY_HTOD       18 // cuMemcpyHtoDAsync_v2
#define CUDA_FUNC_MEMCPY_DTOH       19 // cuMemcpyDtoHAsync_v2
#define CUDA_FUNC_MEMCPY_DTOD       20 // cuMemcpyDtoDAsync_v2
#define CUDA_FUNC_MEMCPY_PEER       21 // cuMemcpyPeer
#define CUDA_FUNC_MEMCPY_PEER_ASYNC 22 // cuMemcpyPeerAsync
#define CUDA_FUNC_MEMSET            23 // cuMemsetD8_v2
#define CUDA_FUNC_MEMSET_ASYNC      24 // cuMemsetD8Async
#define CUDA_FUNC_GRAPH_LAUNCH      25 // cuGraphLaunch (error-only uretprobe; success path stays at entry)

typedef struct cuda_kernel_launch {
    u8 flags; // Must be first, we use it to tell what kind of packet we have on the ring buffer
    u8 _pad[3];
    pid_info pid_info;
    u64 kern_func_off;
    int grid_x;
    int grid_y;
    int grid_z;
    int block_x;
    int block_y;
    int block_z;
    u32 shared_mem_bytes; // Dynamic shared memory bytes (cuLaunchKernel arg 8, cuLaunchKernelEx CUlaunchConfig.sharedMemBytes)
    u32 _pad2;            // align stream_handle on 8 bytes
    u64 stream_handle;    // CUstream handle (cuLaunchKernel arg 9, CUlaunchConfig.hStream); 0 for legacy default stream
} cuda_kernel_launch_t;

typedef struct cuda_malloc {
    u8 flags;    // Must be first, we use it to tell what kind of packet we have on the ring buffer
    u8 mem_kind; // CUDA_MEM_KIND_* — device, host, managed, pool
    u8 _pad[2];
    pid_info pid_info;
    s64 size;
} cuda_malloc_t;

typedef struct cuda_memcpy {
    u8 flags;     // Must be first, we use it to tell what kind of packet we have on the ring buffer
    u8 direction; // CUDA_MEMCPY_DIR_* — HtoD=1, DtoH=2, DtoD=3
    u8 _pad[2];
    pid_info pid_info;
    s64 size;
    u64 stream_handle; // CUstream handle (0 for sync variants and legacy default stream)
} cuda_memcpy_t;

typedef struct cuda_graph_launch {
    u8 flags; // Must be first, we use it to tell what kind of packet we have on the ring buffer
    u8 kind;
    u8 _pad[2];
    pid_info pid_info;
} cuda_graph_launch_t;

// Synchronize event — emitted on exit from cuStreamSynchronize, cuCtxSynchronize, cuEventSynchronize
typedef struct cuda_sync {
    u8 flags;        // Must be first (k_event_sync = 6)
    u8 sync_kind;    // CUDA_SYNC_KIND_*
    u8 _pad[2];
    pid_info pid_info;
    u64 entry_ts;    // bpf_ktime_get_ns() at function entry
    u64 duration_ns; // exit_ts - entry_ts
    s32 retval;      // cudaError_t return value
    s32 _retval_pad; // alignment padding
    u64 handle;      // CUstream / CUcontext / CUevent depending on sync_kind; 0 for cuCtxSynchronize (current ctx not available from BPF)
} cuda_sync_t;

// Free event — emitted on exit (success-only) from cuMemFree_v2 / cuMemFreeHost / cuMemFreeAsync
typedef struct cuda_free {
    u8 flags;     // Must be first (k_event_free = 7)
    u8 mem_kind;  // CUDA_MEM_KIND_*
    u8 _pad[2];
    pid_info pid_info;
    s64 size;          // freed bytes from gpu_alloc_sizes map, 0 if unknown
    u64 stream_handle; // CUstream for cuMemFreeAsync; 0 for sync free
} cuda_free_t;

// Memset event — emitted on exit (success-only) from cuMemsetD8_v2 / cuMemsetD8Async
typedef struct cuda_memset {
    u8 flags;    // Must be first (k_event_memset = 8)
    u8 is_async; // 0=synchronous, 1=asynchronous
    u8 _pad[2];
    pid_info pid_info;
    s64 size;
    u64 stream_handle; // CUstream for async memset; 0 for sync
} cuda_memset_t;

// Peer copy event — emitted on exit (success-only) from cuMemcpyPeer / cuMemcpyPeerAsync
typedef struct cuda_peer_copy {
    u8 flags; // Must be first (k_event_peer_copy = 9)
    u8 _pad[3];
    s32 src_device;
    s32 dst_device;
    pid_info pid_info;
    s64 size;
    u64 stream_handle; // CUstream for cuMemcpyPeerAsync; 0 for sync
} cuda_peer_copy_t;

// Kernel launch done event — emitted on exit from cudaLaunchKernel and cudaLaunchCooperativeKernel
typedef struct cuda_kernel_launch_done {
    u8 flags;        // Must be first (k_event_kernel_launch_done = 5)
    u8 _pad[3];
    pid_info pid_info;
    u64 entry_ts;    // bpf_ktime_get_ns() at function entry
    u64 duration_ns; // exit_ts - entry_ts (host-side launch latency)
    s32 retval;      // cudaError_t return value
    s32 _retval_pad; // alignment padding
} cuda_kernel_launch_done_t;

// Error event — emitted when a CUDA function returns a non-zero cudaError_t
typedef struct cuda_error {
    u8 flags;       // Must be first (k_event_error = 10)
    u8 func_id;     // CUDA_FUNC_* identifier
    u8 _pad[2];
    pid_info pid_info;
    s32 error_code; // cudaError_t value
    s32 _error_pad; // alignment padding
} cuda_error_t;

// Internal BPF map key: scopes a device/host pointer by its owning process (upper 32 bits
// of bpf_get_current_pid_tgid()). Without the tgid component, two concurrent pods can map
// the same virtual pointer value and corrupt each other's alloc/free accounting.
typedef struct cuda_alloc_key {
    u64 ptr;       // device or host pointer value
    u32 tgid;      // upper 32 bits of pid_tgid (process identity)
    u32 _pad;      // explicit padding for -Wpadded
} cuda_alloc_key_t;

// Internal BPF map value: tracks in-flight device allocations for ptr→size lookup on cudaFree
typedef struct cuda_alloc_record {
    s64 size;
    u8  mem_kind;
    u8  _pad[7];
} cuda_alloc_record_t;

// Internal BPF map value: tracks in-flight allocation calls so exit probe can record devPtr
typedef struct cuda_ongoing_alloc {
    u64 devptr_addr; // userspace address of the void** argument (points to result ptr)
    s64 size;
    u8  mem_kind;
    u8  _pad[7];
} cuda_ongoing_alloc_t;

// Internal BPF map value: tracks an in-flight cuMemFree*/cuMemFreeHost call so the
// uretprobe can emit a free event only when the call returns CUDA_SUCCESS.
typedef struct cuda_ongoing_free {
    u64 ptr;            // device or host pointer being freed
    s64 size_hint;      // looked up from gpu_alloc_sizes at entry; 0 if unknown
    u64 stream_handle;  // CUstream for cuMemFreeAsync; 0 otherwise
    u8  mem_kind;       // CUDA_MEM_KIND_* — from alloc record if found, else hinted from caller
    u8  _pad[7];
} cuda_ongoing_free_t;

// Internal BPF map value: tracks an in-flight memcpy call (sync or async).
typedef struct cuda_ongoing_memcpy {
    s64 size;
    u64 stream_handle;  // 0 for sync variants
    u8  direction;      // CUDA_MEMCPY_DIR_*
    u8  _pad[7];
} cuda_ongoing_memcpy_t;

// Internal BPF map value: tracks an in-flight memset call.
typedef struct cuda_ongoing_memset {
    s64 size;
    u64 stream_handle;  // 0 for sync memset
    u8  is_async;
    u8  _pad[7];
} cuda_ongoing_memset_t;

// Internal BPF map value: tracks an in-flight cuMemcpyPeer*/cuMemcpyPeerAsync call.
// CUcontext args are opaque handles; src_device/dst_device exported as 0 until a
// future plan adds context→device resolution.
typedef struct cuda_ongoing_peer_copy {
    s64 size;
    u64 stream_handle;  // 0 for sync peer copy
    s32 src_device;
    s32 dst_device;
} cuda_ongoing_peer_copy_t;

// Internal BPF map value: per-PID in-flight sync call state.
// Captures the entry timestamp and the handle passed to the sync API so the
// uretprobe can report which CUstream / CUevent was waited on.
typedef struct cuda_ongoing_sync {
    u64 entry_ts;
    u64 handle;     // CUstream / CUevent; 0 for cuCtxSynchronize (no arg)
} cuda_ongoing_sync_t;

// Internal BPF map value: per-PID in-flight cuMemHostRegister call.
// Pointer is known at entry (caller-supplied), so we don't need devptr_addr.
typedef struct cuda_ongoing_host_reg {
    u64 ptr;
    s64 size;
} cuda_ongoing_host_reg_t;

// Internal BPF map value: per-PID in-flight cuEventElapsedTime call.
// Saved at entry so the uretprobe can read *pMilliseconds and report the elapsed
// GPU time between hStart and hEnd. This is the ONLY way to capture true GPU
// execution time without CUPTI — the value reflects on-device runtime, not host
// submission latency.
typedef struct cuda_ongoing_event_elapsed {
    u64 p_milliseconds_addr; // userspace address of `float *pMilliseconds`
    u64 h_start;             // CUevent handle
    u64 h_end;               // CUevent handle
} cuda_ongoing_event_elapsed_t;

// Event: cuEventElapsedTime returned successfully.
// elapsed_ms_bits carries the IEEE-754 float32 milliseconds value as a u32 bit
// pattern; userspace decodes via math.Float32frombits and converts to seconds.
typedef struct cuda_event_elapsed {
    u8  flags;             // k_event_event_elapsed = 13
    u8  _pad[3];
    pid_info pid_info;
    u32 elapsed_ms_bits;   // IEEE-754 float32 bit pattern of milliseconds
    u32 _pad2;             // align u64 handles
    u64 h_start;           // CUevent handle (internal — NOT exposed as Prometheus label)
    u64 h_end;             // CUevent handle (internal)
} cuda_event_elapsed_t;

// HAMi OOM event — emitted by uretprobe on libvgpu.so:cuMemAlloc_v2 when rc != 0
// (HAMi quota-denied allocation; never reaches libcuda.so)
typedef struct hami_oom {
    u8  flags;       // k_event_hami_oom = 11
    u8  cuda_func_id; // CUDA_FUNC_* identifier
    u8  mem_kind;    // CUDA_MEM_KIND_* (always DEVICE for cuMemAlloc_v2)
    u8  _pad0;
    s32 rc;          // CUresult error code (CUDA_ERROR_OUT_OF_MEMORY = 2)
    u8  _pad1[4];
    pid_info pid_info;
} hami_oom_t;

// HAMi compute-throttle event — emitted in libcuda.so:cuLaunchKernel entry probe
// when a matching libvgpu.so entry timestamp exists, indicating the HAMi rate_limiter stalled.
// Layout: flags(1) + _pad0[3](3) = 4 bytes; pid_info(12) ends at 16; duration_ns(8) at 16.
typedef struct hami_throttle {
    u8  flags;       // k_event_hami_throttle = 12
    u8  _pad0[3];
    pid_info pid_info;
    u64 duration_ns; // stall time = libcuda_entry_ts - libvgpu_entry_ts
} hami_throttle_t;
