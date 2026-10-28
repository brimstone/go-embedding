package embedding

import (
	"context"
	"errors"
	"strconv"
	"sync/atomic"
	"testing"
)

func newPipeline(t testing.TB, e Embedder, opts PipelineOptions) *Pipeline {
	t.Helper()
	c, err := NewRecursiveChunker(200, 40)
	if err != nil {
		t.Fatal(err)
	}
	p, err := NewPipeline(c, e, opts)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestNewPipelineValidation(t *testing.T) {
	c, _ := NewFixedChunker(10, 0)
	e := newHash(t)
	if _, err := NewPipeline(nil, e, PipelineOptions{}); err == nil {
		t.Error("nil chunker accepted")
	}
	if _, err := NewPipeline(c, nil, PipelineOptions{}); err == nil {
		t.Error("nil embedder accepted")
	}
	if _, err := NewPipeline(c, e, PipelineOptions{Workers: -1}); err == nil {
		t.Error("negative workers accepted")
	}
}

func TestPipelineIngest(t *testing.T) {
	text := genText(3, 10000)
	for _, workers := range []int{1, 4} {
		t.Run("workers="+strconv.Itoa(workers), func(t *testing.T) {
			store := NewMemoryStore()
			e := newHash(t)
			p := newPipeline(t, e, PipelineOptions{Store: store, BatchSize: 3, Workers: workers})
			recs, err := p.Ingest(context.Background(), "doc1", text, map[string]string{"source": "test", MetaText: "overridden"})
			if err != nil {
				t.Fatal(err)
			}
			if len(recs) < 10 {
				t.Fatalf("only %d records", len(recs))
			}
			if store.Len() != len(recs) {
				t.Fatalf("store has %d records, want %d", store.Len(), len(recs))
			}
			for i, r := range recs {
				m := r.Metadata
				if r.ID != "doc1#"+strconv.Itoa(i) || m[MetaDocID] != "doc1" || m[MetaChunk] != strconv.Itoa(i) || m["source"] != "test" {
					t.Fatalf("record %d: id %q meta %v", i, r.ID, m)
				}
				start, _ := strconv.Atoi(m[MetaStart])
				end, _ := strconv.Atoi(m[MetaEnd])
				if text[start:end] != m[MetaText] {
					t.Fatalf("record %d: offsets do not match text", i)
				}
				// Vectors are in chunk order regardless of worker scheduling.
				want := embedOne(t, e, m[MetaText])
				for j := range want {
					if r.Vector[j] != want[j] {
						t.Fatalf("record %d has the wrong vector", i)
					}
				}
				if got, ok := store.Get(r.ID); !ok || got.Metadata[MetaText] != m[MetaText] {
					t.Fatalf("record %d not stored", i)
				}
			}
		})
	}
}

func TestPipelineEmptyText(t *testing.T) {
	store := NewMemoryStore()
	p := newPipeline(t, newHash(t), PipelineOptions{Store: store})
	recs, err := p.Ingest(context.Background(), "d", "  \n ", nil)
	if err != nil || len(recs) != 0 || store.Len() != 0 {
		t.Errorf("recs %v err %v len %d", recs, err, store.Len())
	}
	if _, err := p.Ingest(context.Background(), "", "text", nil); err == nil {
		t.Error("empty doc ID accepted")
	}
}

// countingEmbedder fails on its n-th call.
type countingEmbedder struct {
	inner  Embedder
	calls  atomic.Int32
	failAt int32
}

func (c *countingEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if c.calls.Add(1) == c.failAt {
		return nil, errors.New("embed failed")
	}
	return c.inner.Embed(ctx, texts)
}
func (c *countingEmbedder) Dimension() int { return c.inner.Dimension() }

type failStore struct{}

func (failStore) UpsertBatch([]Record) error { return errors.New("disk full") }

func TestPipelineErrors(t *testing.T) {
	text := genText(4, 5000)
	for _, workers := range []int{1, 4} {
		store := NewMemoryStore()
		e := &countingEmbedder{inner: newHash(t), failAt: 3}
		p := newPipeline(t, e, PipelineOptions{Store: store, BatchSize: 2, Workers: workers})
		if _, err := p.Ingest(context.Background(), "d", text, nil); err == nil || store.Len() != 0 {
			t.Errorf("workers=%d: err %v, stored %d", workers, err, store.Len())
		}
	}

	p := newPipeline(t, newHash(t), PipelineOptions{Store: failStore{}})
	if _, err := p.Ingest(context.Background(), "d", text, nil); err == nil {
		t.Error("store error not returned")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	p = newPipeline(t, newHash(t), PipelineOptions{BatchSize: 1, Workers: 4})
	if _, err := p.Ingest(ctx, "d", text, nil); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

type badEmbedder struct{ vecs [][]float32 }

func (b badEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	return b.vecs[:min(len(b.vecs), len(texts))], nil
}
func (b badEmbedder) Dimension() int { return 0 }

func TestPipelineRejectsBadVectors(t *testing.T) {
	c, _ := NewFixedChunker(5, 0)
	for name, e := range map[string]Embedder{
		"short":    badEmbedder{vecs: [][]float32{{1}}},
		"mismatch": badEmbedder{vecs: [][]float32{{1, 2}, {1}}},
		"empty":    badEmbedder{vecs: [][]float32{{}, {}}},
	} {
		p, _ := NewPipeline(c, e, PipelineOptions{Workers: 1})
		if _, err := p.Ingest(context.Background(), "d", "aaaa bbbb", nil); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestPipelineRetrieval(t *testing.T) {
	docs := map[string]string{
		"pets":     "Dogs are loyal companions and love long walks. Cats are independent and sleep most of the day.",
		"cooking":  "To bake bread, mix flour, water, yeast and salt, knead the dough and let it rise before baking.",
		"database": "A vector database stores embeddings and answers nearest neighbour queries using an index such as HNSW.",
	}
	store := NewMemoryStore()
	p := newPipeline(t, newHash(t), PipelineOptions{Store: store})
	for id, text := range docs {
		if _, err := p.Ingest(context.Background(), id, text, nil); err != nil {
			t.Fatal(err)
		}
	}
	for query, want := range map[string]string{
		"which index does the vector database use": "database",
		"how long should the dough rise":           "cooking",
		"do cats sleep a lot":                      "pets",
	} {
		q, err := p.EmbedQuery(context.Background(), query)
		if err != nil {
			t.Fatal(err)
		}
		hits := store.Search(q, 1)
		if len(hits) != 1 || hits[0].Metadata[MetaDocID] != want {
			t.Errorf("query %q: got %+v, want doc %q", query, hits, want)
		}
	}
}

func TestMemoryStoreCopiesAndSearch(t *testing.T) {
	s := NewMemoryStore()
	v := []float32{1, 0}
	m := map[string]string{"k": "v"}
	_ = s.UpsertBatch([]Record{{ID: "a", Vector: v, Metadata: m}, {ID: "b", Vector: []float32{0, 1}}, {ID: "c", Vector: []float32{1, 1, 1}}})
	v[0], m["k"] = 9, "changed"
	r, _ := s.Get("a")
	if r.Vector[0] != 1 || r.Metadata["k"] != "v" {
		t.Error("store did not copy record")
	}
	hits := s.Search([]float32{1, 0.1}, 5)
	if len(hits) != 2 || hits[0].ID != "a" || hits[1].ID != "b" {
		t.Errorf("hits = %+v", hits)
	}
	if s.Search([]float32{1, 0}, 0) != nil {
		t.Error("k=0 returned hits")
	}
}
