// Package glove bundles GloVe word vectors so that semantic embedding
// works with no files to download or convert:
//
//	e, err := glove.Embedder()
//	if err != nil { ... }
//	p, err := embedding.NewPipeline(chunker, e, embedding.PipelineOptions{Store: store})
//
// The vectors are the 100,000 most frequent words of GloVe 6B, with 300
// dimensions. Importing this package adds about 32 MB to your binary;
// programs that do not import it pay nothing.
//
// # Provenance
//
// The data is glove.6B.300d.txt from glove.6B.zip (SHA-256
// 6471382cdd837544bf3ac72497a38715e845897d265b2b424b4761832009c837),
// published by Stanford NLP at https://nlp.stanford.edu/projects/glove/
// and mirrored at https://huggingface.co/stanfordnlp/glove. It was
// converted with
//
//	go run ./cmd/glove-convert -in glove.6B.zip -member glove.6B.300d.txt -out glove/glove-300d-100k.wv
//
// which is deterministic and reproduces glove-300d-100k.wv exactly
// (SHA-256 c15a1372b0d11f4c255d6752458f73fca153fca7540b6a635b0fc0aa3fdc7cda).
//
// # License
//
// GloVe 6B is released under the Open Data Commons Public Domain
// Dedication and License (PDDL) 1.0,
// https://opendatacommons.org/licenses/pddl/1-0/. Citing Pennington,
// Socher and Manning, "GloVe: Global Vectors for Word Representation",
// EMNLP 2014, is appreciated.
package glove

import (
	"bytes"
	_ "embed"
	"sync"

	embedding "github.com/brimstone/go-embedding"
)

//go:embed glove-300d-100k.wv
var data []byte

// Dimension is the length of the vectors returned by Embedder.
const Dimension = 300

var load = sync.OnceValues(func() (*embedding.WordVectors, error) {
	return embedding.ReadWordVectors(bytes.NewReader(data))
})

// WordVectors returns the bundled word vectors. They are decoded on the
// first call, which takes about 0.1 s and 15–40 MB of memory; later calls
// return the same value. An error means the binary itself is damaged.
func WordVectors() (*embedding.WordVectors, error) { return load() }

// Embedder returns a VectorEmbedder over the bundled vectors with default
// SIF options. Use WordVectors with embedding.NewVectorEmbedder for other
// options.
func Embedder() (*embedding.VectorEmbedder, error) {
	wv, err := load()
	if err != nil {
		return nil, err
	}
	return embedding.NewVectorEmbedder(wv, embedding.SIFOptions{})
}
