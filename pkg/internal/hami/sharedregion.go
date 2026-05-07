// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package hami reads HAMi-core shared region cache files (cudevshr.cache).
// The binary layout matches HAMi-core major_version=1, minor_version=1
// (HAMi-core/src/multiprocess/multiprocess_memory_limit.h, MAJOR_VERSION 1 MINOR_VERSION 1).
//
// Layout source:
//   shared_region_t  → sharedRegionV1
//   shrreg_proc_slot_t → procSlotV1
//   device_memory_t   → deviceMemoryV1
//   device_util_t     → deviceUtilV1
package hami

import (
	"fmt"
	"os"
	"strings"
	"syscall"
	"unsafe"
)

const (
	MagicFlag  = 19920718
	MajorV1    = 1
	maxDevices = 16
	maxProcs   = 1024
)

// deviceMemoryV1 mirrors device_memory_t (64 bytes).
type deviceMemoryV1 struct {
	contextSize uint64
	moduleSize  uint64
	bufferSize  uint64 // called data_size in some versions
	offset      uint64
	total       uint64
	_           [3]uint64 // unused
}

// deviceUtilV1 mirrors device_util_t (48 bytes).
type deviceUtilV1 struct {
	decUtil uint64
	encUtil uint64
	smUtil  uint64
	_       [3]uint64 // unused
}

// procSlotV1 mirrors shrreg_proc_slot_t (1960 bytes).
type procSlotV1 struct {
	pid         int32
	hostpid     int32
	used        [maxDevices]deviceMemoryV1
	monitorused [maxDevices]uint64
	deviceUtil  [maxDevices]deviceUtilV1
	status      int32
	_           [3]uint64 // unused + alignment padding
}

// uuidV1 is a 96-byte null-terminated C string.
type uuidV1 struct {
	raw [96]byte
}

func (u uuidV1) String() string {
	s := string(u.raw[:])
	if idx := strings.IndexByte(s, 0); idx >= 0 {
		return s[:idx]
	}
	return s
}

// semV1 is the glibc sem_t (32 bytes on x86-64).
type semV1 struct {
	_ [32]byte
}

// sharedRegionV1 mirrors shared_region_t (major=1 minor=1).
// Field ownerPid is uint32 to match the Go v1 struct in HAMi monitor package
// (confirmed offset 56 for device_num via both uint32+padding and size_t paths).
type sharedRegionV1 struct {
	initializedFlag   int32
	majorVersion      int32
	minorVersion      int32
	smInitFlag        int32
	ownerPid          uint32
	sem               semV1
	deviceNum         uint64
	uuids             [maxDevices]uuidV1
	limit             [maxDevices]uint64
	smLimit           [maxDevices]uint64
	procs             [maxProcs]procSlotV1
	procNum           int32
	utilizationSwitch int32
	recentKernel      int32
	priority          int32
	lastKernelTime    int64
	_                 [4]uint64 // unused
}

// DeviceSample is a snapshot of one GPU device's metrics for one container.
type DeviceSample struct {
	DeviceIndex  int
	UUID         string
	MemLimitBytes uint64
	SMLimit       uint64
}

// ProcSample is a snapshot of one process slot.
type ProcSample struct {
	PID     int32
	HostPID int32
	Status  int32
	// Per-device slices (len == DeviceCount from ContainerSample)
	ContextBytes  []uint64
	ModuleBytes   []uint64
	BufferBytes   []uint64
	TotalBytes    []uint64
	MonitorBytes  []uint64
	SMUtil        []uint64
	EncUtil       []uint64
	DecUtil       []uint64
}

// ContainerSample is a full snapshot of one container's cache file.
type ContainerSample struct {
	PodUID        string
	ContainerName string
	FilePath      string
	DeviceCount   int
	Devices       []DeviceSample // len == DeviceCount
	Procs         []ProcSample   // only slots where pid > 0
}

// SharedRegion is an open, mmap'd HAMi cache file.
// The underlying *os.File is kept open for the mmap lifetime.
type SharedRegion struct {
	filePath string
	data     []byte
	f        *os.File // kept open so the mmap pages remain valid
	sr       *sharedRegionV1
}

// OpenSharedRegion mmaps a HAMi cudevshr.cache file and validates the magic flag.
// The caller must call Close() when done.
func OpenSharedRegion(path string) (*SharedRegion, error) {
	f, err := os.OpenFile(path, os.O_RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}

	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}

	size := int(info.Size())
	minSize := int(unsafe.Sizeof(sharedRegionV1{}))
	if size < minSize {
		f.Close()
		return nil, fmt.Errorf("%s: too small (%d < %d bytes)", path, size, minSize)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}

	sr := (*sharedRegionV1)(unsafe.Pointer(&data[0]))

	if sr.initializedFlag != MagicFlag {
		got := sr.initializedFlag // capture before unmap
		syscall.Munmap(data)
		f.Close()
		return nil, fmt.Errorf("%s: bad magic %d (want %d)", path, got, MagicFlag)
	}
	if sr.majorVersion != MajorV1 {
		got := sr.majorVersion // capture before unmap
		syscall.Munmap(data)
		f.Close()
		return nil, fmt.Errorf("%s: unsupported major version %d (want %d)", path, got, MajorV1)
	}

	return &SharedRegion{
		filePath: path,
		data:     data,
		f:        f,
		sr:       sr,
	}, nil
}

// Close unmaps the shared region and closes the backing file.
func (s *SharedRegion) Close() {
	if s.data != nil {
		syscall.Munmap(s.data)
		s.data = nil
	}
	if s.f != nil {
		s.f.Close()
		s.f = nil
	}
}

// Snapshot reads all fields into a ContainerSample. Safe to call repeatedly.
func (s *SharedRegion) Snapshot(podUID, containerName string) ContainerSample {
	sr := s.sr
	devCount := int(sr.deviceNum)
	if devCount > maxDevices {
		devCount = maxDevices
	}

	devices := make([]DeviceSample, devCount)
	for d := 0; d < devCount; d++ {
		devices[d] = DeviceSample{
			DeviceIndex:   d,
			UUID:          sr.uuids[d].String(),
			MemLimitBytes: sr.limit[d],
			SMLimit:       sr.smLimit[d],
		}
	}

	procCount := int(sr.procNum)
	if procCount < 0 || procCount > maxProcs {
		procCount = maxProcs
	}

	procs := make([]ProcSample, 0, procCount)
	for i := 0; i < procCount; i++ {
		p := &sr.procs[i]
		if p.pid == 0 && p.hostpid == 0 {
			continue
		}
		ps := ProcSample{
			PID:          p.pid,
			HostPID:      p.hostpid,
			Status:       p.status,
			ContextBytes: make([]uint64, devCount),
			ModuleBytes:  make([]uint64, devCount),
			BufferBytes:  make([]uint64, devCount),
			TotalBytes:   make([]uint64, devCount),
			MonitorBytes: make([]uint64, devCount),
			SMUtil:       make([]uint64, devCount),
			EncUtil:      make([]uint64, devCount),
			DecUtil:      make([]uint64, devCount),
		}
		for d := 0; d < devCount; d++ {
			ps.ContextBytes[d] = p.used[d].contextSize
			ps.ModuleBytes[d] = p.used[d].moduleSize
			ps.BufferBytes[d] = p.used[d].bufferSize
			ps.TotalBytes[d] = p.used[d].total
			ps.MonitorBytes[d] = p.monitorused[d]
			ps.SMUtil[d] = p.deviceUtil[d].smUtil
			ps.EncUtil[d] = p.deviceUtil[d].encUtil
			ps.DecUtil[d] = p.deviceUtil[d].decUtil
		}
		procs = append(procs, ps)
	}

	return ContainerSample{
		PodUID:        podUID,
		ContainerName: containerName,
		FilePath:      s.filePath,
		DeviceCount:   devCount,
		Devices:       devices,
		Procs:         procs,
	}
}
