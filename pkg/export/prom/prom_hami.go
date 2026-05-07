// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package prom

import (
	"context"
	"strconv"

	"github.com/prometheus/client_golang/prometheus"

	"go.opentelemetry.io/obi/pkg/internal/hami"
)

// hamiGauges holds all HAMi Prometheus metrics registered by newHamiGauges.
type hamiGauges struct {
	// Per-container, per-device quota limits set by HAMi at pod start.
	quotaMemLimitBytes *prometheus.GaugeVec
	quotaSMLimitPct    *prometheus.GaugeVec

	// Per-process, per-device metrics (NVML-measured, updated ~1s/120ms).
	procMemContextBytes *prometheus.GaugeVec
	procMemModuleBytes  *prometheus.GaugeVec
	procMemBufferBytes  *prometheus.GaugeVec
	procMemTotalBytes   *prometheus.GaugeVec
	procMemNVMLBytes    *prometheus.GaugeVec
	procSMUtilPct       *prometheus.GaugeVec
	procEncUtilPct      *prometheus.GaugeVec
	procDecUtilPct      *prometheus.GaugeVec
	procStatus          *prometheus.GaugeVec

}

var quotaLabels = []string{"pod_uid", "container_name", "device_id", "gpu_uuid"}
var procLabels = []string{"pod_uid", "container_name", "device_id", "gpu_uuid", "pid"}

func newHamiGauges() *hamiGauges {
	return &hamiGauges{
		quotaMemLimitBytes: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_hami_quota_memory_limit_bytes",
			Help: "HAMi memory quota limit assigned to this container per GPU device",
		}, quotaLabels),
		quotaSMLimitPct: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_hami_quota_sm_limit_percent",
			Help: "HAMi SM compute quota (%) assigned to this container per GPU device",
		}, quotaLabels),
		procMemContextBytes: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_hami_proc_memory_context_bytes",
			Help: "GPU context memory (driver overhead) used by this process",
		}, procLabels),
		procMemModuleBytes: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_hami_proc_memory_module_bytes",
			Help: "GPU module/kernel-code memory loaded by this process",
		}, procLabels),
		procMemBufferBytes: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_hami_proc_memory_buffer_bytes",
			Help: "GPU data/buffer memory (cudaMalloc) used by this process",
		}, procLabels),
		procMemTotalBytes: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_hami_proc_memory_total_bytes",
			Help: "Total GPU memory used by this process (context+module+buffer)",
		}, procLabels),
		procMemNVMLBytes: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_hami_proc_memory_nvml_bytes",
			Help: "NVML-reported physical GPU memory used by this process (~1s cadence)",
		}, procLabels),
		procSMUtilPct: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_hami_proc_sm_utilization_percent",
			Help: "NVML SM utilization % for this process (~120ms cadence from HAMi watcher)",
		}, procLabels),
		procEncUtilPct: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_hami_proc_enc_utilization_percent",
			Help: "NVML video encoder utilization % for this process",
		}, procLabels),
		procDecUtilPct: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_hami_proc_dec_utilization_percent",
			Help: "NVML video decoder utilization % for this process",
		}, procLabels),
		procStatus: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gpu_hami_proc_status",
			Help: "HAMi process suspension state (0=running, non-zero=suspended by compute throttle)",
		}, procLabels),
	}
}

// collectors returns all GaugeVecs as prometheus.Collector for registration.
func (g *hamiGauges) collectors() []prometheus.Collector {
	return []prometheus.Collector{
		g.quotaMemLimitBytes,
		g.quotaSMLimitPct,
		g.procMemContextBytes,
		g.procMemModuleBytes,
		g.procMemBufferBytes,
		g.procMemTotalBytes,
		g.procMemNVMLBytes,
		g.procSMUtilPct,
		g.procEncUtilPct,
		g.procDecUtilPct,
		g.procStatus,
	}
}

// update applies one ContainerSample to all gauges. Called from the poller goroutine.
func (g *hamiGauges) update(s hami.ContainerSample) {
	for _, dev := range s.Devices {
		devID := strconv.Itoa(dev.DeviceIndex)
		qLabels := prometheus.Labels{
			"pod_uid":        s.PodUID,
			"container_name": s.ContainerName,
			"device_id":      devID,
			"gpu_uuid":       dev.UUID,
		}
		g.quotaMemLimitBytes.With(qLabels).Set(float64(dev.MemLimitBytes))
		g.quotaSMLimitPct.With(qLabels).Set(float64(dev.SMLimit))
	}

	for _, proc := range s.Procs {
		pidStr := strconv.Itoa(int(proc.HostPID))
		for d, dev := range s.Devices {
			pLabels := prometheus.Labels{
				"pod_uid":        s.PodUID,
				"container_name": s.ContainerName,
				"device_id":      strconv.Itoa(dev.DeviceIndex),
				"gpu_uuid":       dev.UUID,
				"pid":            pidStr,
			}
			g.procMemContextBytes.With(pLabels).Set(float64(proc.ContextBytes[d]))
			g.procMemModuleBytes.With(pLabels).Set(float64(proc.ModuleBytes[d]))
			g.procMemBufferBytes.With(pLabels).Set(float64(proc.BufferBytes[d]))
			g.procMemTotalBytes.With(pLabels).Set(float64(proc.TotalBytes[d]))
			g.procMemNVMLBytes.With(pLabels).Set(float64(proc.MonitorBytes[d]))
			g.procSMUtilPct.With(pLabels).Set(float64(proc.SMUtil[d]))
			g.procEncUtilPct.With(pLabels).Set(float64(proc.EncUtil[d]))
			g.procDecUtilPct.With(pLabels).Set(float64(proc.DecUtil[d]))
			g.procStatus.With(pLabels).Set(float64(proc.Status))
		}
	}
}

// startHamiPoller starts the HAMi cache poller goroutine if GPU metrics are enabled.
// containerDir should be the hostPath-mounted HAMi containers directory.
func startHamiPoller(ctx context.Context, g *hamiGauges, containerDir string) {
	if g == nil {
		return
	}
	p := hami.NewPoller(containerDir, 0)
	go func() {
		p.Run(ctx, func(s hami.ContainerSample) {
			g.update(s)
		})
	}()
}

