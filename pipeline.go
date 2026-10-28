package embedding

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"runtime"
	"strconv"
	"sync"
)

// Metadata keys set on every Record produced by a Pipeline.
const (
	MetaDocID = "doc_id" // the document ID passed to Ingest
	MetaChunk = "chunk"  // the chunk index within the document
	MetaStart = "start"  // byte offset of the chunk in the document
	MetaEnd   = "end"    // byte offset of the end of the chunk
	MetaText  = "text"   // the chunk text
)

// Record is an embedded chunk, shaped like a vector database record.
type Record struct {
	// ID is "<docID>#<chunk index>".
	ID       string
	Vector   []float32
	Metadata map[string]string
}

// Store receives the records produced by a Pipeline. A vector database
// collection can be adapted to it in a few lines.
type Store interface {
	// UpsertBatch inserts or replaces the given records. Implementations
	// should copy anything they retain.
	UpsertBatch(recs []Record) error
}

// PipelineOptions configures a Pipeline.
type PipelineOptions struct {
	// Store, if non-nil, receives the records of each ingested document in
	// a single UpsertBatch call.
	Store Store
	// BatchSize is the number of chunks per Embed call. Default 32.
	BatchSize int
	// Workers is the number of concurrent Embed calls. Default GOMAXPROCS.
	Workers int
}

// Pipeline chunks text, embeds the chunks and stores the results.
// It is safe for concurrent use if its Chunker, Embedder and Store are.
type Pipeline struct {
	chunker   Chunker
	embedder  Embedder
	store     Store
	batchSize int
	workers   int
}

// NewPipeline returns a Pipeline using c to chunk and e to embed.
func NewPipeline(c Chunker, e Embedder, opts PipelineOptions) (*Pipeline, error) {
	if c == nil || e == nil {
		return nil, errors.New("embedding: chunker and embedder are required")
	}
	if opts.BatchSize < 0 || opts.Workers < 0 {
		return nil, errors.New("embedding: negative BatchSize or Workers")
	}
	if opts.BatchSize == 0 {
		opts.BatchSize = 32
	}
	if opts.Workers == 0 {
		opts.Workers = runtime.GOMAXPROCS(0)
	}
	return &Pipeline{chunker: c, embedder: e, store: opts.Store, batchSize: opts.BatchSize, workers: opts.Workers}, nil
}

// Ingest chunks text, embeds every chunk and, if the pipeline has a Store,
// upserts the records. meta is copied into every record's metadata
// alongside the Meta* keys, which take precedence. It returns the records
// in chunk order; a text with no content yields no records.
func (p *Pipeline) Ingest(ctx context.Context, docID, text string, meta map[string]string) ([]Record, error) {
	if docID == "" {
		return nil, errors.New("embedding: empty document ID")
	}
	chunks := p.chunker.Chunk(text)
	if len(chunks) == 0 {
		return nil, nil
	}
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	vecs, err := p.embedAll(ctx, texts)
	if err != nil {
		return nil, err
	}
	recs := make([]Record, len(chunks))
	for i, c := range chunks {
		m := make(map[string]string, len(meta)+5)
		maps.Copy(m, meta)
		m[MetaDocID] = docID
		m[MetaChunk] = strconv.Itoa(c.Index)
		m[MetaStart] = strconv.Itoa(c.Start)
		m[MetaEnd] = strconv.Itoa(c.End)
		m[MetaText] = c.Text
		recs[i] = Record{ID: docID + "#" + strconv.Itoa(c.Index), Vector: vecs[i], Metadata: m}
	}
	if p.store != nil {
		if err := p.store.UpsertBatch(recs); err != nil {
			return nil, fmt.Errorf("embedding: storing %q: %w", docID, err)
		}
	}
	return recs, nil
}

// EmbedQuery embeds a single search query with the pipeline's Embedder.
func (p *Pipeline) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	vs, err := p.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(vs) != 1 {
		return nil, fmt.Errorf("embedding: embedder returned %d vectors for 1 input", len(vs))
	}
	return vs[0], nil
}

// embedAll embeds texts in batches using up to p.workers goroutines.
func (p *Pipeline) embedAll(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	nb := (len(texts) + p.batchSize - 1) / p.batchSize
	embed := func(ctx context.Context, b int) error {
		lo, hi := b*p.batchSize, min((b+1)*p.batchSize, len(texts))
		vs, err := p.embedder.Embed(ctx, texts[lo:hi])
		if err != nil {
			return fmt.Errorf("embedding: embedding chunks %d-%d: %w", lo, hi-1, err)
		}
		if len(vs) != hi-lo {
			return fmt.Errorf("embedding: embedder returned %d vectors for %d inputs", len(vs), hi-lo)
		}
		copy(out[lo:hi], vs)
		return nil
	}

	if workers := min(p.workers, nb); workers <= 1 {
		for b := range nb {
			if err := embed(ctx, b); err != nil {
				return nil, err
			}
		}
	} else {
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		var (
			wg       sync.WaitGroup
			once     sync.Once
			firstErr error
		)
		jobs := make(chan int)
		for range workers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for b := range jobs {
					if err := embed(ctx, b); err != nil {
						once.Do(func() { firstErr = err; cancel() })
					}
				}
			}()
		}
	feed:
		for b := range nb {
			select {
			case jobs <- b:
			case <-ctx.Done():
				break feed
			}
		}
		close(jobs)
		wg.Wait()
		if firstErr != nil {
			return nil, firstErr
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}

	dim := len(out[0])
	for i, v := range out {
		if len(v) == 0 || len(v) != dim {
			return nil, fmt.Errorf("embedding: chunk %d has %d dimensions, want %d", i, len(v), dim)
		}
	}
	return out, nil
}
