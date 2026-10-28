package embedding

import (
	"context"
	"errors"
	"math"
	"testing"
)

func newHash(t testing.TB) *HashEmbedder {
	t.Helper()
	e, err := NewHashEmbedder(DefaultHashOptions())
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func embedOne(t testing.TB, e Embedder, s string) []float32 {
	t.Helper()
	vs, err := e.Embed(context.Background(), []string{s})
	if err != nil {
		t.Fatal(err)
	}
	return vs[0]
}

func TestHashEmbedderValidation(t *testing.T) {
	for _, o := range []HashOptions{{Dimension: 0}, {Dimension: 8, MinN: -1}, {Dimension: 8, MinN: 4, MaxN: 3}} {
		if _, err := NewHashEmbedder(o); err == nil {
			t.Errorf("NewHashEmbedder(%+v) succeeded", o)
		}
	}
}

func TestHashEmbedderShapeAndNorm(t *testing.T) {
	e := newHash(t)
	vs, err := e.Embed(context.Background(), []string{"hello world", "", "!!! ???", "another text"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vs) != 4 {
		t.Fatalf("got %d vectors", len(vs))
	}
	for i, v := range vs {
		if len(v) != e.Dimension() {
			t.Fatalf("vector %d has len %d", i, len(v))
		}
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		wantZero := i == 1 || i == 2
		if wantZero && sum != 0 {
			t.Errorf("vector %d: want zero vector for text without words", i)
		}
		if !wantZero && math.Abs(sum-1) > 1e-5 {
			t.Errorf("vector %d: squared norm %v, want 1", i, sum)
		}
	}
	// Appending to one vector must not clobber the next.
	if cap(vs[0]) != len(vs[0]) {
		t.Errorf("vector capacity %d exceeds length %d", cap(vs[0]), len(vs[0]))
	}
}

func TestHashEmbedderDeterministicAndCaseInsensitive(t *testing.T) {
	e := newHash(t)
	a := embedOne(t, e, "The Quick Brown Fox")
	b := embedOne(t, e, "the quick brown fox")
	c := embedOne(t, newHash(t), "  the, quick... brown\tfox!")
	for i := range a {
		if a[i] != b[i] || a[i] != c[i] {
			t.Fatalf("embeddings differ at %d: %v %v %v", i, a[i], b[i], c[i])
		}
	}
}

func TestHashEmbedderSimilarity(t *testing.T) {
	e := newHash(t)
	q := embedOne(t, e, "how do I store vectors in a database")
	near := embedOne(t, e, "storing vector embeddings in a vector database")
	far := embedOne(t, e, "my cat enjoys sleeping in the afternoon sun")
	if sn, sf := Cosine(q, near), Cosine(q, far); sn <= sf {
		t.Errorf("similar text scored %v, unrelated %v", sn, sf)
	}
	// Character n-grams make typos and inflections close.
	typo := embedOne(t, e, "databse")
	right := embedOne(t, e, "database")
	other := embedOne(t, e, "elephant")
	if Cosine(typo, right) <= Cosine(typo, other) {
		t.Error("typo not closer to its correct spelling")
	}
}

func TestHashEmbedderNoCharNGrams(t *testing.T) {
	e, err := NewHashEmbedder(HashOptions{Dimension: 64})
	if err != nil {
		t.Fatal(err)
	}
	if c := Cosine(embedOne(t, e, "database"), embedOne(t, e, "databse")); c > 0.5 {
		t.Errorf("without char n-grams different words have cosine %v", c)
	}
}

func TestHashEmbedderWithIDF(t *testing.T) {
	corpus := []string{
		"the cat sat on the mat", "the dog ate the bone", "the bird flew over the tree",
		"the fish swam in the sea", "the cow ate the grass",
	}
	plain := newHash(t)
	idf := plain.WithIDF(corpus)
	a, b := "the cat and the mat", "the fish and the sea"
	if p, w := Cosine(embedOne(t, plain, a), embedOne(t, plain, b)), Cosine(embedOne(t, idf, a), embedOne(t, idf, b)); w >= p {
		t.Errorf("IDF did not reduce similarity driven by common words: plain %v, idf %v", p, w)
	}
	if plain.idf != nil {
		t.Error("WithIDF modified the receiver")
	}
}

func TestHashEmbedderContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := newHash(t).Embed(ctx, []string{"x"}); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestCosine(t *testing.T) {
	if c := Cosine([]float32{1, 0}, []float32{1, 0}); math.Abs(float64(c)-1) > 1e-6 {
		t.Errorf("identical: %v", c)
	}
	if c := Cosine([]float32{1, 0}, []float32{0, 1}); c != 0 {
		t.Errorf("orthogonal: %v", c)
	}
	if c := Cosine([]float32{1, 2}, []float32{-1, -2}); math.Abs(float64(c)+1) > 1e-6 {
		t.Errorf("opposite: %v", c)
	}
	if c := Cosine([]float32{0, 0}, []float32{1, 2}); c != 0 {
		t.Errorf("zero: %v", c)
	}
	defer func() {
		if recover() == nil {
			t.Error("length mismatch did not panic")
		}
	}()
	Cosine([]float32{1}, []float32{1, 2})
}
