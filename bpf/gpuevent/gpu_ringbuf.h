// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bpfcore/utils.h>

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 23); // 8 MiB — accommodates entry+exit probes for free/memcpy/memset/peer
                                  // and larger cuda_kernel_launch_t (64B) under high-rate bursts
                                  // such as inference loops or PyTorch DataLoader workers.
} gpu_events SEC(".maps");
