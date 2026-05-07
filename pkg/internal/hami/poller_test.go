// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package hami

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
	"unsafe"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// makeCacheFile writes a synthetic sharedRegionV1 to a temp file and returns its path.
// The struct is zeroed then specific fields are set; the file is exactly sizeof(sharedRegionV1).
func makeCacheFile(t *testing.T, dir string, mutate func(*sharedRegionV1)) string {
	t.Helper()
	var region sharedRegionV1
	region.initializedFlag = MagicFlag
	region.majorVersion = MajorV1
	region.minorVersion = 1
	if mutate != nil {
		mutate(&region)
	}
	size := int(unsafe.Sizeof(region))
	data := unsafe.Slice((*byte)(unsafe.Pointer(&region)), size)

	path := filepath.Join(dir, "test.cache")
	require.NoError(t, os.WriteFile(path, data, 0600))
	return path
}

// TestOpenSharedRegion_BadMagic ensures OpenSharedRegion rejects wrong magic.
func TestOpenSharedRegion_BadMagic(t *testing.T) {
	dir := t.TempDir()
	var region sharedRegionV1
	region.initializedFlag = 0xDEAD // wrong
	size := int(unsafe.Sizeof(region))
	data := unsafe.Slice((*byte)(unsafe.Pointer(&region)), size)
	path := filepath.Join(dir, "bad.cache")
	require.NoError(t, os.WriteFile(path, data, 0600))

	_, err := OpenSharedRegion(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "bad magic")
}

// TestOpenSharedRegion_UnsupportedVersion rejects major version != 1.
func TestOpenSharedRegion_UnsupportedVersion(t *testing.T) {
	dir := t.TempDir()
	path := makeCacheFile(t, dir, func(r *sharedRegionV1) {
		r.majorVersion = 99
	})
	_, err := OpenSharedRegion(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported major version")
}

// TestSnapshot_QuotaFields reads memory/SM limit from a synthetic file.
func TestSnapshot_QuotaFields(t *testing.T) {
	dir := t.TempDir()
	path := makeCacheFile(t, dir, func(r *sharedRegionV1) {
		r.deviceNum = 1
		r.uuids[0].raw[0] = 'G'
		r.uuids[0].raw[1] = 'P'
		r.uuids[0].raw[2] = 'U'
		r.uuids[0].raw[3] = 0 // null-terminate
		r.limit[0] = 8 * 1024 * 1024 * 1024   // 8 GiB
		r.smLimit[0] = 50
	})

	sr, err := OpenSharedRegion(path)
	require.NoError(t, err)
	defer sr.Close()

	sample := sr.Snapshot("pod-uid-abc", "mycontainer")
	assert.Equal(t, "pod-uid-abc", sample.PodUID)
	assert.Equal(t, "mycontainer", sample.ContainerName)
	assert.Equal(t, 1, sample.DeviceCount)
	require.Len(t, sample.Devices, 1)
	assert.Equal(t, "GPU", sample.Devices[0].UUID)
	assert.Equal(t, uint64(8*1024*1024*1024), sample.Devices[0].MemLimitBytes)
	assert.Equal(t, uint64(50), sample.Devices[0].SMLimit)
	assert.Empty(t, sample.Procs) // procNum=0 → no procs
}

// TestSnapshot_ProcFields reads per-process metrics from a synthetic file.
func TestSnapshot_ProcFields(t *testing.T) {
	dir := t.TempDir()
	path := makeCacheFile(t, dir, func(r *sharedRegionV1) {
		r.deviceNum = 1
		r.uuids[0].raw[0] = 'A'
		r.uuids[0].raw[1] = 0
		r.limit[0] = 4 * 1024 * 1024 * 1024
		r.smLimit[0] = 30

		r.procNum = 1
		r.procs[0].pid = 1001
		r.procs[0].hostpid = 2002
		r.procs[0].status = 0
		r.procs[0].used[0].contextSize = 100
		r.procs[0].used[0].moduleSize = 200
		r.procs[0].used[0].bufferSize = 300
		r.procs[0].used[0].total = 600
		r.procs[0].monitorused[0] = 700
		r.procs[0].deviceUtil[0].smUtil = 42
		r.procs[0].deviceUtil[0].encUtil = 5
		r.procs[0].deviceUtil[0].decUtil = 3
	})

	sr, err := OpenSharedRegion(path)
	require.NoError(t, err)
	defer sr.Close()

	sample := sr.Snapshot("pod-uid-xyz", "trainer")
	require.Len(t, sample.Procs, 1)
	p := sample.Procs[0]
	assert.Equal(t, int32(1001), p.PID)
	assert.Equal(t, int32(2002), p.HostPID)
	assert.Equal(t, int32(0), p.Status)
	assert.Equal(t, uint64(100), p.ContextBytes[0])
	assert.Equal(t, uint64(200), p.ModuleBytes[0])
	assert.Equal(t, uint64(300), p.BufferBytes[0])
	assert.Equal(t, uint64(600), p.TotalBytes[0])
	assert.Equal(t, uint64(700), p.MonitorBytes[0])
	assert.Equal(t, uint64(42), p.SMUtil[0])
	assert.Equal(t, uint64(5), p.EncUtil[0])
	assert.Equal(t, uint64(3), p.DecUtil[0])
}

// TestSnapshot_NoProcWhenPIDZero ensures zero-pid slots are skipped.
func TestSnapshot_NoProcWhenPIDZero(t *testing.T) {
	dir := t.TempDir()
	path := makeCacheFile(t, dir, func(r *sharedRegionV1) {
		r.deviceNum = 1
		r.procNum = 1
		// procs[0].pid = 0 (zero value) → should be skipped
	})
	sr, err := OpenSharedRegion(path)
	require.NoError(t, err)
	defer sr.Close()
	sample := sr.Snapshot("p", "c")
	assert.Empty(t, sample.Procs)
}

// TestParseDirName covers valid and invalid directory name formats.
func TestParseDirName(t *testing.T) {
	cases := []struct {
		input         string
		wantPodUID    string
		wantContainer string
		wantOK        bool
	}{
		{
			input:         "4d3f2e1a-0000-0000-0000-000000000001_mycontainer",
			wantPodUID:    "4d3f2e1a-0000-0000-0000-000000000001",
			wantContainer: "mycontainer",
			wantOK:        true,
		},
		{
			input:         "4d3f2e1a-0000-0000-0000-000000000001_gpu-worker_extra",
			wantPodUID:    "4d3f2e1a-0000-0000-0000-000000000001",
			wantContainer: "gpu-worker_extra",
			wantOK:        true,
		},
		{input: "tooshort", wantOK: false},
		{input: "nodashesatall00000000000000000000000", wantOK: false},
	}
	for _, tc := range cases {
		uid, cname, ok := parseDirName(tc.input)
		assert.Equal(t, tc.wantOK, ok, "input=%q", tc.input)
		if tc.wantOK {
			assert.Equal(t, tc.wantPodUID, uid, "input=%q", tc.input)
			assert.Equal(t, tc.wantContainer, cname, "input=%q", tc.input)
		}
	}
}

// TestPoller_EmitsOneCacheFile verifies the poller discovers and emits a sample
// from a synthetic container directory.
func TestPoller_EmitsOneCacheFile(t *testing.T) {
	baseDir := t.TempDir()

	// Create {podUID}_{containerName}/ dir with a .cache file
	podUID := "aaaabbbb-cccc-dddd-eeee-ffffffffffff"
	containerName := "trainer"
	contDir := filepath.Join(baseDir, podUID+"_"+containerName)
	require.NoError(t, os.MkdirAll(contDir, 0755))
	makeCacheFile(t, contDir, func(r *sharedRegionV1) {
		r.deviceNum = 1
		r.limit[0] = 1024
		r.smLimit[0] = 25
	})

	p := NewPoller(baseDir, 50*time.Millisecond)
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	var got []ContainerSample
	done := make(chan struct{})
	go func() {
		p.Run(ctx, func(s ContainerSample) {
			got = append(got, s)
			if len(got) >= 1 {
				cancel()
			}
		})
		close(done)
	}()
	<-done

	require.NotEmpty(t, got)
	s := got[0]
	assert.Equal(t, podUID, s.PodUID)
	assert.Equal(t, containerName, s.ContainerName)
	assert.Equal(t, 1, s.DeviceCount)
	assert.Equal(t, uint64(1024), s.Devices[0].MemLimitBytes)
	assert.Equal(t, uint64(25), s.Devices[0].SMLimit)
}

// TestPoller_EmptyDirIsNoOp ensures no panic and no emission when dir is empty.
func TestPoller_EmptyDirIsNoOp(t *testing.T) {
	baseDir := t.TempDir()
	p := NewPoller(baseDir, 50*time.Millisecond)
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Millisecond)
	defer cancel()

	var count int
	p.Run(ctx, func(ContainerSample) { count++ })
	assert.Equal(t, 0, count)
}

// TestPoller_MissingDirIsNoOp ensures no panic when the base dir does not exist.
func TestPoller_MissingDirIsNoOp(t *testing.T) {
	p := NewPoller("/nonexistent/path/that/does/not/exist", 50*time.Millisecond)
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Millisecond)
	defer cancel()

	var count int
	p.Run(ctx, func(ContainerSample) { count++ })
	assert.Equal(t, 0, count)
}
