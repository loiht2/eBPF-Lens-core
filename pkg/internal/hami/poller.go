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

func (p *Poller) poll(emit func(ContainerSample)) {
	entries, err := os.ReadDir(p.containerDir)
	if err != nil {
		if !os.IsNotExist(err) {
			plog().Warn("cannot read HAMi container dir", "path", p.containerDir, "err", err)
		}
		return
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		podUID, containerName, ok := parseDirName(entry.Name())
		if !ok {
			continue
		}
		dirPath := filepath.Join(p.containerDir, entry.Name())
		p.pollDir(dirPath, podUID, containerName, emit)
	}
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

// parseDirName splits "{podUID}_{containerName}" into its two parts.
// The pod UID is a standard UUID (contains hyphens); the first underscore
// after the UUID is the separator.
//
// Format: {36-char UUID}_{containerName}
// Example: 4d3f2e1a-0000-0000-0000-000000000001_mycontainer
func parseDirName(name string) (podUID, containerName string, ok bool) {
	// Pod UIDs are 36 chars (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx).
	// Find the first underscore after position 36.
	if len(name) < 38 { // 36 + '_' + at least 1 char
		return "", "", false
	}
	idx := strings.Index(name[36:], "_")
	if idx < 0 {
		return "", "", false
	}
	sep := 36 + idx
	return name[:sep], name[sep+1:], true
}
