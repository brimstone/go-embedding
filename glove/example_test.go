package glove_test

import (
	"context"
	"fmt"
	"log"

	embedding "github.com/brimstone/go-embedding"
	"github.com/brimstone/go-embedding/glove"
)

func Example() {
	e, err := glove.Embedder()
	if err != nil {
		log.Fatal(err)
	}
	store := embedding.NewMemoryStore()
	chunker, _ := embedding.NewRecursiveChunker(200, 0)
	p, _ := embedding.NewPipeline(chunker, e, embedding.PipelineOptions{Store: store})

	ctx := context.Background()
	p.Ingest(ctx, "garage", "Our mechanics service every make of car, from oil changes to new brakes.", nil)
	p.Ingest(ctx, "bakery", "Fresh sourdough loaves and pastries are baked every morning.", nil)

	q, _ := p.EmbedQuery(ctx, "where can I get my automobile fixed?")
	fmt.Println(store.Search(q, 1)[0].Metadata[embedding.MetaDocID])

	wv, _ := glove.WordVectors()
	nb, _ := wv.MostSimilar("automobile", 3)
	for _, n := range nb {
		fmt.Println(n.Word)
	}
	// Output:
	// garage
	// auto
	// automobiles
	// automotive
}
