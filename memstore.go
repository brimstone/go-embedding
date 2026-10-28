package embedding

import (
	"maps"
	"slices"
	"sort"
	"sync"
)

// MemoryStore is an in-memory Store with exact cosine-similarity search.
// It is intended for tests and small prototypes. It is safe for concurrent
// use.
type MemoryStore struct {
	mu   sync.RWMutex
	recs map[string]Record
}

// Hit is a MemoryStore search result.
type Hit struct {
	Record
	// Score is the cosine similarity to the query; larger is closer.
	Score float32
}

// NewMemoryStore returns an empty MemoryStore.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{recs: make(map[string]Record)}
}

// UpsertBatch implements Store. Records are copied.
func (s *MemoryStore) UpsertBatch(recs []Record) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, r := range recs {
		s.recs[r.ID] = Record{ID: r.ID, Vector: slices.Clone(r.Vector), Metadata: maps.Clone(r.Metadata)}
	}
	return nil
}

// Len returns the number of stored records.
func (s *MemoryStore) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.recs)
}

// Get returns the record with the given ID.
func (s *MemoryStore) Get(id string) (Record, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	r, ok := s.recs[id]
	return r, ok
}

// Search returns the k records most similar to query, best first. Records
// whose vector length differs from the query are skipped. The returned
// records share memory with the store and must not be modified.
func (s *MemoryStore) Search(query []float32, k int) []Hit {
	if k <= 0 {
		return nil
	}
	s.mu.RLock()
	hits := make([]Hit, 0, len(s.recs))
	for _, r := range s.recs {
		if len(r.Vector) == len(query) {
			hits = append(hits, Hit{Record: r, Score: Cosine(query, r.Vector)})
		}
	}
	s.mu.RUnlock()
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		return hits[i].ID < hits[j].ID
	})
	return hits[:min(k, len(hits))]
}
