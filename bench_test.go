package embedding

import (
	"bytes"
	"context"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
)

var benchText = genText(42, 1<<20) // ~1 MiB

func BenchmarkChunk(b *testing.B) {
	for _, size := range []int{256, 1024} {
		for name, c := range chunkers(b, size, size/8) {
			b.Run(name+"/size="+strconv.Itoa(size), func(b *testing.B) {
				b.SetBytes(int64(len(benchText)))
				b.ReportAllocs()
				for range b.N {
					c.Chunk(benchText)
				}
			})
		}
	}
}

func benchChunks(b *testing.B, size int) []string {
	b.Helper()
	c, _ := NewRecursiveChunker(size, 0)
	chunks := c.Chunk(benchText)
	texts := make([]string, len(chunks))
	for i, ch := range chunks {
		texts[i] = ch.Text
	}
	return texts
}

func BenchmarkHashEmbed(b *testing.B) {
	texts := benchChunks(b, 512)[:256]
	var n int64
	for _, t := range texts {
		n += int64(len(t))
	}
	opts := map[string]HashOptions{
		"default":     DefaultHashOptions(),
		"words-only":  {Dimension: 512},
		"dim=1536":    {Dimension: 1536, MinN: 3, MaxN: 5, WordBigrams: true},
		"ngrams=3..6": {Dimension: 512, MinN: 3, MaxN: 6, WordBigrams: true},
	}
	for name, o := range opts {
		e, err := NewHashEmbedder(o)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(name, func(b *testing.B) {
			b.SetBytes(n)
			b.ReportAllocs()
			for range b.N {
				if _, err := e.Embed(context.Background(), texts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkHashEmbedSingleQuery(b *testing.B) {
	e := newHash(b)
	q := []string{"how do I store vector embeddings in a database?"}
	b.ReportAllocs()
	for range b.N {
		if _, err := e.Embed(context.Background(), q); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWithIDF(b *testing.B) {
	e := newHash(b)
	corpus := benchChunks(b, 512)[:1000]
	b.ReportAllocs()
	for range b.N {
		e.WithIDF(corpus)
	}
}

func BenchmarkPipelineIngest(b *testing.B) {
	for _, workers := range []int{1, 4, 0} {
		name := "workers=" + strconv.Itoa(workers)
		if workers == 0 {
			name = "workers=GOMAXPROCS"
		}
		b.Run(name, func(b *testing.B) {
			p := newPipeline(b, newHash(b), PipelineOptions{Workers: workers, Store: NewMemoryStore()})
			b.SetBytes(int64(len(benchText)))
			b.ReportAllocs()
			for range b.N {
				if _, err := p.Ingest(context.Background(), "doc", benchText, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkHandler(b *testing.B) {
	h := &Handler{Models: map[string]Embedder{"hash": newHash(b)}}
	texts := benchChunks(b, 512)[:32]
	var sb strings.Builder
	sb.WriteString(`{"model":"hash","input":[`)
	for i, t := range texts {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(strconv.Quote(t))
	}
	sb.WriteString(`]}`)
	body := sb.String()
	for _, format := range []string{"float", "base64"} {
		req := strings.Replace(body, `{"model"`, `{"encoding_format":"`+format+`","model"`, 1)
		b.Run(format, func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				rec := httptest.NewRecorder()
				h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(req)))
				if rec.Code != http.StatusOK {
					b.Fatal(rec.Body.String())
				}
			}
		})
	}
}

func BenchmarkMemoryStoreSearch(b *testing.B) {
	store := NewMemoryStore()
	p := newPipeline(b, newHash(b), PipelineOptions{Store: store})
	if _, err := p.Ingest(context.Background(), "doc", benchText[:256<<10], nil); err != nil {
		b.Fatal(err)
	}
	q, _ := p.EmbedQuery(context.Background(), "vector database index")
	b.ReportMetric(float64(store.Len()), "records")
	b.ReportAllocs()
	for range b.N {
		store.Search(q, 10)
	}
}

// benchVectors is a synthetic 100k × 300 table, the size of a typical
// converted GloVe file. Set EMBEDDING_GLOVE to benchmark a real one.
var benchVectorsOnce = sync.OnceValue(func() *WordVectors {
	if path := os.Getenv("EMBEDDING_GLOVE"); path != "" {
		if wv, err := LoadWordVectors(path); err == nil {
			return wv
		}
	}
	const n, d = 100000, 300
	rng := rand.New(rand.NewSource(1))
	words := make([]string, n)
	x := make([]float32, n*d)
	prob := make([]float32, n)
	for i := range words {
		words[i] = vocabWord(i)
		prob[i] = float32(1 / ((float64(i+1) + 2.7) * 12.5))
	}
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	return quantize(d, words, prob, x)
})

// vocabWord makes the benchmark's vocabulary match benchText's words.
func vocabWord(i int) string {
	if i < len(vocab) {
		return vocab[i]
	}
	return "w" + strconv.Itoa(i)
}

func BenchmarkVectorEmbed(b *testing.B) {
	e, err := NewVectorEmbedder(benchVectorsOnce(), SIFOptions{})
	if err != nil {
		b.Fatal(err)
	}
	texts := benchChunks(b, 512)[:256]
	var n int64
	for _, t := range texts {
		n += int64(len(t))
	}
	b.SetBytes(n)
	b.ReportAllocs()
	for range b.N {
		if _, err := e.Embed(context.Background(), texts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVectorEmbedSingleQuery(b *testing.B) {
	e, _ := NewVectorEmbedder(benchVectorsOnce(), SIFOptions{})
	q := []string{"how do I store vector embeddings in a database?"}
	b.ReportAllocs()
	for range b.N {
		if _, err := e.Embed(context.Background(), q); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLoadWordVectors(b *testing.B) {
	var buf bytes.Buffer
	if _, err := benchVectorsOnce().WriteTo(&buf); err != nil {
		b.Fatal(err)
	}
	data := buf.Bytes()
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	for range b.N {
		if _, err := ReadWordVectors(bytes.NewReader(data)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMostSimilar(b *testing.B) {
	wv := benchVectorsOnce()
	word := wv.Word(500)
	b.ReportAllocs()
	for range b.N {
		wv.MostSimilar(word, 10)
	}
}

func BenchmarkReadGloVe(b *testing.B) {
	text, _, _ := randomGloVe(1, 10000, 300)
	b.SetBytes(int64(len(text)))
	b.ReportAllocs()
	for range b.N {
		if _, err := ReadGloVe(strings.NewReader(text), GloVeOptions{}); err != nil {
			b.Fatal(err)
		}
	}
}
