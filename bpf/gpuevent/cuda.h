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
} cuda_memcpy_t;

typedef struct cuda_graph_launch {
    u8 flags; // Must be first, we use it to tell what kind of packet we have on the ring buffer
    u8 kind;
    u8 _pad[2];
    pid_info pid_info;
} cuda_graph_launch_t;

// Synchronize event — emitted on exit from cudaStreamSynchronize, cudaDeviceSynchronize, cudaEventSynchronize
typedef struct cuda_sync {
    u8 flags;        // Must be first (k_event_sync = 6)
    u8 sync_kind;    // CUDA_SYNC_KIND_*
    u8 _pad[2];
    pid_info pid_info;
    u64 entry_ts;    // bpf_ktime_get_ns() at function entry
    u64 duration_ns; // exit_ts - entry_ts
    s32 retval;      // cudaError_t return value
    s32 _retval_pad; // alignment padding
} cuda_sync_t;

// Free event — emitted on entry to cudaFree, cudaFreeHost, cudaFreeAsync
typedef struct cuda_free {
    u8 flags;     // Must be first (k_event_free = 7)
    u8 mem_kind;  // CUDA_MEM_KIND_*
    u8 _pad[2];
    pid_info pid_info;
    s64 size;     // freed bytes from gpu_alloc_sizes map, -1 if unknown
} cuda_free_t;

// Memset event — emitted on entry to cudaMemset and cudaMemsetAsync
typedef struct cuda_memset {
    u8 flags;    // Must be first (k_event_memset = 8)
    u8 is_async; // 0=synchronous, 1=asynchronous
    u8 _pad[2];
    pid_info pid_info;
    s64 size;
} cuda_memset_t;

// Peer copy event — emitted on entry to cudaMemcpyPeer and cudaMemcpyPeerAsync (multi-GPU)
typedef struct cuda_peer_copy {
    u8 flags; // Must be first (k_event_peer_copy = 9)
    u8 _pad[3];
    s32 src_device;
    s32 dst_device;
    pid_info pid_info;
    s64 size;
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
