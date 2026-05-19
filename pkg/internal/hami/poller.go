// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package hami

import (
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	// DefaultContainerDir is where HAMi writes per-container cache files.
	// Both device-plugin and DRA modes use {HOOK_PATH}/vgpu/containers/.
	// Mount this path as a hostPath volume in the DaemonSet.
	DefaultContainerDir = "/usr/local/vgpu/containers"

	DefaultPollInterval = time.Second
)

func plog() *slog.Logger {
	return slog.With("component", "hami-poller")
}

// Poller discovers HAMi cudevshr.cache files under a base directory and
// emits ContainerSample snapshots on every poll interval.
//
// Directory structure expected:
//
//	{containerDir}/{podUID}_{containerName}/*.cache
type Poller struct {
	containerDir string
	interval     time.Duration
}

// NewPoller creates a Poller. containerDir is typically DefaultContainerDir.
func NewPoller(containerDir string, interval time.Duration) *Poller {
	if containerDir == "" {
		containerDir = DefaultContainerDir
	}
	if interval <= 0 {
		interval = DefaultPollInterval
	}
	return &Poller{
		containerDir: containerDir,
		interval:     interval,
	}
}

// Run starts the polling loop. Calls emit for each ContainerSample found on
// every tick. Blocks until ctx is cancelled.
func (p *Poller) Run(ctx context.Context, emit func(ContainerSample)) {
	ticker := time.NewTicker(p.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.poll(emit)
		}
	}
}

// effectiveContainerDir returns the first accessible path for containerDir,
// trying the direct path first, then /proc/1/root prefix.
// The agent DaemonSet may not mount the HAMi directory as a volume, but with
// hostPID=true + privileged=true the host filesystem is reachable via /proc/1/root.
func (p *Poller) effectiveContainerDir() string {
	if _, err := os.Stat(p.containerDir); err == nil {
		return p.containerDir
	}
	hostPath := "/proc/1/root" + p.containerDir
	if _, err := os.Stat(hostPath); err == nil {
		return hostPath
	}
	return p.containerDir // return original so caller gets a meaningful error
}

func (p *Poller) poll(emit func(ContainerSample)) {
	dir := p.effectiveContainerDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		if !os.IsNotExist(err) {
			plog().Warn("cannot read HAMi container dir", "path", dir, "err", err)
		}
		return
	}

	// Build a fresh PID→Binding snapshot while emitting samples.
	// Register all process-device pairs seen in the cache file — a process
	// appears after cuInit, before any memory allocation, so filtering by
	// memory usage would drop the binding before the first BPF event fires.
	snapshot := make(map[int32][]Binding, 256)

	wrappedEmit := func(s ContainerSample) {
		for _, proc := range s.Procs {
			for d, dev := range s.Devices {
				b := Binding{
					PodUID:        s.PodUID,
					ContainerName: s.ContainerName,
					DeviceIndex:   d,
					GPUUUID:       dev.UUID,
				}
				snapshot[proc.HostPID] = append(snapshot[proc.HostPID], b)
			}
		}
		emit(s)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		podUID, containerName, ok := parseDirName(entry.Name())
		if !ok {
			continue
		}
		dirPath := filepath.Join(dir, entry.Name())
		p.pollDir(dirPath, podUID, containerName, wrappedEmit)
	}

	DefaultPIDIndex().Replace(snapshot)
}

// pollDir scans one {podUID}_{containerName}/ directory for .cache files
// and emits a snapshot for each valid one.
func (p *Poller) pollDir(dirPath, podUID, containerName string, emit func(ContainerSample)) {
	files, err := os.ReadDir(dirPath)
	if err != nil {
		return
	}
	for _, f := range files {
		if f.IsDir() || !strings.HasSuffix(f.Name(), ".cache") {
			continue
		}
		cachePath := filepath.Join(dirPath, f.Name())
		sr, err := OpenSharedRegion(cachePath)
		if err != nil {
			plog().Debug("skipping cache file", "path", cachePath, "err", err)
			continue
		}
		sample := sr.Snapshot(podUID, containerName)
		sr.Close()
		emit(sample)
	}
}

// parseDirName parses the container directory name into (podUID, containerName).
//
// Two formats are supported:
//
//	Format A (device-plugin mode): {36-char UUID}_{containerName}
//	  Example: 4d3f2e1a-0000-0000-0000-000000000001_mycontainer
//
//	Format B (DRA mode): {36-char UUID}  (ResourceClaim UID, no container suffix)
//	  Example: e9fd2003-f1b3-495e-b6a6-79451e88c61f
//
// For Format B the returned containerName is "".
func parseDirName(name string) (podUID, containerName string, ok bool) {
	if len(name) < 36 {
		return "", "", false
	}
	// Validate UUID shape at fixed hyphen positions.
	if name[8] != '-' || name[13] != '-' || name[18] != '-' || name[23] != '-' {
		return "", "", false
	}
	if len(name) == 36 {
		// Format B: plain UUID (DRA mode — dir is the ResourceClaim UID).
		return name, "", true
	}
	if name[36] != '_' || len(name) < 38 {
		return "", "", false
	}
	return name[:36], name[37:], true
}
