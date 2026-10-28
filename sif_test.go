package embedding

import (
	"context"
	"errors"
	"hash/crc32"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func crcIEEE(b []byte) uint32 { return crc32.ChecksumIEEE(b) }

// toyVectors is a small vocabulary where synonyms share a direction.
// Earlier words are treated as more frequent.
func toyVectors(t testing.TB) *WordVectors {
	t.Helper()
	words := []string{"the", "a", "is", "do", "n't", "'s", "it", "e-mail",
		"car", "automobile", "vehicle", "repair", "fix", "banana", "fruit", "eat", "mail", "send"}
	vecs := [][]float32{
		{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 0.9}, {1, 1, 1, 1, 0.9, 1}, {0.2, 0, 0, 0, 0, 1}, {-1, 0, 0, 0, 0, 0}, {0, 0, 0, 0.1, 0, 0}, {0, 0.1, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0.5},
		{1, 0, 0, 0, 0, 0}, {0.95, 0.1, 0, 0, 0, 0}, {0.9, 0.2, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0}, {0, 0.95, 0.1, 0, 0, 0},
		{0, 0, 1, 0, 0, 0}, {0, 0, 0.9, 0.2, 0, 0}, {0, 0, 0.3, 1, 0, 0},
		{0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0.9, 0.3},
	}
	wv := mustGloVe(t, gloveText(words, vecs), GloVeOptions{NoPostProcessing: true})
	// Zipf estimates over 18 words make every word look common; use
	// probabilities typical of a real 400k-word vocabulary instead.
	for i, w := range words {
		wv.prob[i] = 1e-4
		if i < 8 && w != "e-mail" {
			wv.prob[i] = 0.02
		}
	}
	return wv
}

func toyEmbedder(t testing.TB) *VectorEmbedder {
	t.Helper()
	e, err := NewVectorEmbedder(toyVectors(t), SIFOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestForEachWord(t *testing.T) {
	var got []string
	forEachWord("Don't e-mail the U.S.A. -- it’s state-of-the-art! 'quoted' x-", nil, func(w []byte) { got = append(got, string(w)) })
	want := []string{"don't", "e-mail", "the", "u", "s", "a", "it's", "state-of-the-art", "quoted", "x"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %q\nwant %q", got, want)
	}
}

func TestVectorEmbedderSynonyms(t *testing.T) {
	e := toyEmbedder(t)
	q := embedOne(t, e, "automobile repair")
	near := embedOne(t, e, "how to fix a car")
	far := embedOne(t, e, "eat a banana")
	if sn, sf := Cosine(q, near), Cosine(q, far); sn < 0.9 || sf > 0.3 {
		t.Errorf("paraphrase similarity %v, unrelated %v", sn, sf)
	}
	// HashEmbedder cannot see this: no shared words or n-grams.
	h := newHash(t)
	if c := Cosine(embedOne(t, h, "automobile repair"), embedOne(t, h, "fix car")); c > 0.2 {
		t.Errorf("hash embedder unexpectedly similar: %v", c)
	}
}

func TestVectorEmbedderFrequentWordsDownweighted(t *testing.T) {
	e := toyEmbedder(t)
	// "the" (rank 1) points along all axes; it must barely move "car".
	if c := Cosine(embedOne(t, e, "the the the car"), embedOne(t, e, "car")); c < 0.95 {
		t.Errorf("frequent words dominated: cosine %v", c)
	}
	plain, _ := NewVectorEmbedder(e.wv, SIFOptions{A: 1e6}) // ≈ unweighted average
	if c := Cosine(embedOne(t, plain, "the the the car"), embedOne(t, plain, "car")); c > 0.9 {
		t.Errorf("unweighted average should be dominated by 'the': cosine %v", c)
	}
}

func TestVectorEmbedderLookupSplitting(t *testing.T) {
	e := toyEmbedder(t)
	var got []string
	collect := func(s string) []string {
		got = nil
		forEachWord(s, nil, func(w []byte) {
			e.lookup(w, func(i int32) { got = append(got, e.wv.Word(int(i))) })
		})
		return got
	}
	for in, want := range map[string][]string{
		"e-mail":      {"e-mail"},        // whole hyphenated word in vocabulary
		"car-repair":  {"car", "repair"}, // split on hyphen
		"don't":       {"do", "n't"},     // contraction
		"it's":        {"it", "'s"},
		"zzz":         nil,
		"car-zzz-fix": {"car", "fix"},
	} {
		if g := collect(in); !reflect.DeepEqual(g, want) {
			t.Errorf("%q -> %q, want %q", in, g, want)
		}
	}
}

func TestVectorEmbedderShape(t *testing.T) {
	e := toyEmbedder(t)
	vs, err := e.Embed(context.Background(), []string{"car", "", "unknown words only", "Car!"})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vs {
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		zero := i == 1 || i == 2
		if zero != (sum == 0) || (!zero && math.Abs(sum-1) > 1e-5) {
			t.Errorf("vector %d squared norm %v", i, sum)
		}
		if len(v) != e.Dimension() || cap(v) != len(v) {
			t.Errorf("vector %d len %d cap %d", i, len(v), cap(v))
		}
	}
	if !reflect.DeepEqual(vs[0], vs[3]) {
		t.Error("case or punctuation changed the embedding")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := e.Embed(ctx, []string{"car"}); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v", err)
	}
	if _, err := NewVectorEmbedder(nil, SIFOptions{}); err == nil {
		t.Error("nil vectors accepted")
	}
	if _, err := NewVectorEmbedder(e.wv, SIFOptions{A: -1}); err == nil {
		t.Error("negative A accepted")
	}
}

func TestVectorEmbedderCommonComponent(t *testing.T) {
	e := toyEmbedder(t)
	corpus := []string{"fix the car", "eat the banana", "send the mail", "repair a vehicle", "fruit is food", ""}
	pe := e.WithCommonComponent(corpus)
	if pe.pc == nil || e.pc != nil {
		t.Fatal("component not fitted, or receiver modified")
	}
	for _, s := range corpus[:5] {
		v := embedOne(t, pe, s)
		var p float32
		for j := range v {
			p += v[j] * pe.pc[j]
		}
		if math.Abs(float64(p)) > 1e-4 {
			t.Errorf("%q still has %v along the common component", s, p)
		}
	}
	if empty := e.WithCommonComponent(nil); empty.pc != nil {
		t.Error("component fitted from empty corpus")
	}
}

func TestVectorEmbedderConcurrent(t *testing.T) {
	e := toyEmbedder(t)
	want := embedOne(t, e, "fix my automobile")
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				vs, _ := e.Embed(context.Background(), []string{"fix my automobile"})
				if !reflect.DeepEqual(vs[0], want) {
					t.Error("concurrent embedding differs")
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestVectorEmbedderInPipelineAndHandler(t *testing.T) {
	e := toyEmbedder(t)
	store := NewMemoryStore()
	p := newPipeline(t, e, PipelineOptions{Store: store})
	for id, text := range map[string]string{"cars": "Fix the car.", "food": "Eat a banana."} {
		if _, err := p.Ingest(context.Background(), id, text, nil); err != nil {
			t.Fatal(err)
		}
	}
	q, _ := p.EmbedQuery(context.Background(), "automobile repair")
	if hits := store.Search(q, 1); hits[0].Metadata[MetaDocID] != "cars" {
		t.Errorf("got %s", hits[0].ID)
	}
	code, r := post(t, &Handler{Models: map[string]Embedder{"glove": e}}, `{"input":"automobile"}`)
	if code != 200 || len(r.Data) != 1 || !strings.Contains(string(r.Data[0].Embedding), ",") {
		t.Errorf("handler: %d %+v", code, r.Error)
	}
}
