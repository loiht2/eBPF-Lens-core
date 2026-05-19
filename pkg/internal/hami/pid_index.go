// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package hami

import "sync"

// Binding holds the HAMi device identity for one process-device pair.
type Binding struct {
	PodUID        string
	ContainerName string
	DeviceIndex   int
	GPUUUID       string
}

// PIDIndex maps host PIDs to their HAMi device bindings as observed in the
// most recent cache-file poll. It is the bridge between BPF events (PID-keyed)
// and HAMi instance identity (gpu.uuid-keyed).
//
// One PID may be bound to multiple GPUs (multi-vGPU workloads): callers
// receive the full slice. The first entry is used when a single value is needed.
type PIDIndex struct {
	mu       sync.RWMutex
	bindings map[int32][]Binding
}

// NewPIDIndex returns an empty PIDIndex.
func NewPIDIndex() *PIDIndex {
	return &PIDIndex{bindings: make(map[int32][]Binding, 256)}
}

// Replace atomically swaps the entire index with a new snapshot.
// Called once per poll tick by the poller goroutine.
func (p *PIDIndex) Replace(snapshot map[int32][]Binding) {
	p.mu.Lock()
	p.bindings = snapshot
	p.mu.Unlock()
}

// Lookup returns the bindings for the given host PID, or nil if unknown.
// Safe for concurrent callers.
func (p *PIDIndex) Lookup(hostPID int32) []Binding {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.bindings[hostPID]
}

// LookupUUID returns the GPU UUID for the given host PID.
// When a PID is bound to multiple GPUs, the first binding's UUID is returned
// (deterministic, consistent with the multi-vGPU first-binding policy in plan-4 §3).
// Returns "" when the PID is not found.
func (p *PIDIndex) LookupUUID(hostPID int32) string {
	bs := p.Lookup(hostPID)
	if len(bs) == 0 {
		return ""
	}
	return bs[0].GPUUUID
}

// defaultPIDIndex is the process-scoped singleton shared between the HAMi
// cache poller (writer) and the gpuevent tracer (reader). Both components
// live in different parts of the component graph, so a package-level var
// avoids threading it through unrelated constructors.
var defaultPIDIndex = NewPIDIndex()

// DefaultPIDIndex returns the process-scoped PIDIndex singleton.
func DefaultPIDIndex() *PIDIndex {
	return defaultPIDIndex
}
