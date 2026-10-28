package glove_test

import (
	"context"
	"testing"

	embedding "github.com/brimstone/go-embedding"
	"github.com/brimstone/go-embedding/glove"
)

func TestBundledVectors(t *testing.T) {
	wv, err := glove.WordVectors()
	if err != nil {
		t.Fatal(err)
	}
	if wv.Dim() != glove.Dimension || wv.Len() != 100000 || wv.Word(0) != "the" {
		t.Fatalf("dim %d len %d first %q", wv.Dim(), wv.Len(), wv.Word(0))
	}
	if again, _ := glove.WordVectors(); again != wv {
		t.Error("WordVectors decoded twice")
	}
	related, _ := wv.Similarity("car", "automobile")
	unrelated, _ := wv.Similarity("car", "banana")
	if related < 0.5 || unrelated > 0.2 {
		t.Errorf("car/automobile %.3f, car/banana %.3f", related, unrelated)
	}
}

func TestEmbedder(t *testing.T) {
	e, err := glove.Embedder()
	if err != nil {
		t.Fatal(err)
	}
	if e.Dimension() != glove.Dimension {
		t.Fatalf("dimension %d", e.Dimension())
	}
	vs, err := e.Embed(context.Background(), []string{
		"automobile repair shop",
		"a mechanic who fixes cars",
		"a recipe for banana bread",
	})
	if err != nil {
		t.Fatal(err)
	}
	if near, far := embedding.Cosine(vs[0], vs[1]), embedding.Cosine(vs[0], vs[2]); near < far+0.2 {
		t.Errorf("paraphrase %.3f not clearly closer than unrelated %.3f", near, far)
	}
}
