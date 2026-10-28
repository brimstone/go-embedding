// Package embedding turns blocks of text into vector embeddings using only
// the Go standard library.
//
// The work is split into three pieces:
//
//   - A Chunker breaks a source text into smaller, contiguous Chunks.
//     RecursiveChunker splits on a hierarchy of natural boundaries
//     (paragraphs, lines, sentences, words) and is the recommended default;
//     FixedChunker produces windows of a fixed rune length. Every Chunk
//     records its byte offsets, so source[c.Start:c.End] == c.Text.
//
//   - An Embedder turns texts into fixed-length float32 vectors.
//     HashEmbedder is a fast, deterministic, dependency-free embedder based
//     on feature hashing of words, word bigrams and character n-grams
//     (optionally IDF weighted). It captures lexical similarity, not
//     meaning. VectorEmbedder averages pretrained GloVe word vectors with
//     SIF weighting, so texts about a "car" and an "automobile" are close.
//     The glove subpackage bundles ready-to-use vectors
//     (glove.Embedder()); cmd/glove-convert converts others. Any other
//     model can be plugged in by implementing Embedder.
//
//   - A Pipeline chunks a document, embeds the chunks concurrently in
//     batches and writes the resulting Records to a Store. A Record has the
//     same shape as a vector database record: an ID, a vector and string
//     metadata (including the chunk text and its offsets).
//
// Handler exposes any set of Embedders as an OpenAI-compatible
// POST /v1/embeddings endpoint that programs can mount on their own
// http.ServeMux, so existing OpenAI client libraries can use them.
//
// MemoryStore is a small brute-force Store useful for tests and prototypes.
package embedding
