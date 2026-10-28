package embedding_test

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/brimstone/go-embedding"
)

func Example() {
	chunker, err := embedding.NewRecursiveChunker(80, 0)
	if err != nil {
		log.Fatal(err)
	}
	embedder, err := embedding.NewHashEmbedder(embedding.DefaultHashOptions())
	if err != nil {
		log.Fatal(err)
	}
	store := embedding.NewMemoryStore()
	p, err := embedding.NewPipeline(chunker, embedder, embedding.PipelineOptions{Store: store})
	if err != nil {
		log.Fatal(err)
	}

	text := "Go is a statically typed, compiled language designed at Google.\n\n" +
		"A vector database stores embeddings and finds their nearest neighbours.\n\n" +
		"Bread is made from flour, water, salt and yeast."
	recs, err := p.Ingest(context.Background(), "notes", text, map[string]string{"author": "me"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("chunks:", len(recs))

	q, err := p.EmbedQuery(context.Background(), "which database finds nearest neighbours?")
	if err != nil {
		log.Fatal(err)
	}
	best := store.Search(q, 1)[0]
	fmt.Println(best.ID, best.Metadata[embedding.MetaText])
	// Output:
	// chunks: 3
	// notes#1 A vector database stores embeddings and finds their nearest neighbours.
}

func ExampleRecursiveChunker() {
	c, err := embedding.NewRecursiveChunker(30, 0)
	if err != nil {
		log.Fatal(err)
	}
	for _, ch := range c.Chunk("One sentence here. Another one follows. And a third.") {
		fmt.Printf("%d [%d:%d] %q\n", ch.Index, ch.Start, ch.End, ch.Text)
	}
	// Output:
	// 0 [0:18] "One sentence here."
	// 1 [19:39] "Another one follows."
	// 2 [40:52] "And a third."
}

func ExampleHandler() {
	embedder, err := embedding.NewHashEmbedder(embedding.DefaultHashOptions())
	if err != nil {
		log.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.Handle("POST /v1/embeddings", &embedding.Handler{
		Models: map[string]embedding.Embedder{"hash-512": embedder},
	})
	// Any OpenAI-compatible client can now use base URL http://<host>/v1
	// with model "hash-512".
	_ = http.ListenAndServe // e.g. http.ListenAndServe(":8080", mux)
}
