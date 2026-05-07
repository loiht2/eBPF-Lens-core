// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build obi_bpf_ignore

// HAMi libvgpu.so uprobes — only-HAMi mode.
//
// In only-MIG mode libvgpu.so is absent; these probes never attach and the
// hami_launch_entry map stays empty, so no hami_throttle events are ever emitted.
// No runtime mode flag is needed.

// ──────────── k_event enum extensions (must match cuda.c) ────────────
// k_event_hami_oom      = 11
// k_event_hami_throttle = 12  (emitted from cuda.c, not here)

// ───────────────────────── BPF maps ──────────────────────────────────
// hami_launch_entry (pid_tgid → timestamp) is declared in cuda.c so that
// cu_launch_entry_impl can read it. Probes below write to it.

// ─────────────────── Launch entry probes on libvgpu.so ───────────────

// Record the timestamp when HAMi's shim begins processing a kernel launch.
// The HAMi rate_limiter() runs during this window before calling libcuda.so.
static __always_inline int hami_launch_entry_impl(void *ctx) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&hami_launch_entry, &id, &ts, BPF_ANY);
    return 0;
}

SEC("uprobe/hami_cuLaunchKernel")
int BPF_KPROBE(obi_hami_cu_launch) {
    bpf_dbg_printk("=== uprobe/hami:cuLaunchKernel id=%llx ===", bpf_get_current_pid_tgid());
    return hami_launch_entry_impl(ctx);
}

SEC("uprobe/hami_cuLaunchCooperativeKernel")
int BPF_KPROBE(obi_hami_cu_coop_launch) {
    bpf_dbg_printk("=== uprobe/hami:cuLaunchCooperativeKernel id=%llx ===", bpf_get_current_pid_tgid());
    return hami_launch_entry_impl(ctx);
}

// ─────────────────── Alloc exit probes on libvgpu.so ─────────────────

// When HAMi's quota enforcement rejects an allocation it returns an error code
// (CUDA_ERROR_OUT_OF_MEMORY = 2) without ever calling libcuda.so. The libcuda
// uretprobe therefore never fires, so we must detect it here.
static __always_inline int hami_alloc_exit_impl(struct pt_regs *ctx, u8 cuda_func_id, u8 mem_kind) {
    const u64 id = bpf_get_current_pid_tgid();
    if (!valid_pid(id)) {
        return 0;
    }
    s32 rc = (s32)(long)PT_REGS_RC(ctx);
    if (rc == 0) {
        return 0; // successful alloc — libcuda probes handle this
    }
    hami_oom_t *e = bpf_ringbuf_reserve(&gpu_events, sizeof(*e), 0);
    if (!e) {
        return 0;
    }
    e->flags        = 11; // k_event_hami_oom
    e->cuda_func_id = cuda_func_id;
    e->mem_kind     = mem_kind;
    e->rc           = rc;
    task_pid(&e->pid_info);
    bpf_ringbuf_submit(e, 0);
    return 0;
}

SEC("uretprobe/hami_cuMemAlloc_v2")
int BPF_KRETPROBE(obi_hami_cu_mem_alloc_exit) {
    return hami_alloc_exit_impl(ctx, CUDA_FUNC_MALLOC, CUDA_MEM_KIND_DEVICE);
}

SEC("uretprobe/hami_cuMemAllocManaged")
int BPF_KRETPROBE(obi_hami_cu_mem_alloc_managed_exit) {
    return hami_alloc_exit_impl(ctx, CUDA_FUNC_MANAGED_MALLOC, CUDA_MEM_KIND_MANAGED);
}

SEC("uretprobe/hami_cuMemAllocHost_v2")
int BPF_KRETPROBE(obi_hami_cu_mem_alloc_host_exit) {
    return hami_alloc_exit_impl(ctx, CUDA_FUNC_HOST_MALLOC, CUDA_MEM_KIND_HOST);
}

SEC("uretprobe/hami_cuMemHostAlloc")
int BPF_KRETPROBE(obi_hami_cu_mem_host_alloc_exit) {
    return hami_alloc_exit_impl(ctx, CUDA_FUNC_HOST_ALLOC, CUDA_MEM_KIND_HOST);
}

SEC("uretprobe/hami_cuMemAllocAsync")
int BPF_KRETPROBE(obi_hami_cu_mem_alloc_async_exit) {
    return hami_alloc_exit_impl(ctx, CUDA_FUNC_ASYNC_MALLOC, CUDA_MEM_KIND_DEVICE);
}
