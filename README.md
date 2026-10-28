# embedding

A pure Go library for turning text into vector embeddings. It uses only the standard library: no cgo, no dependencies.

It does three things:

1. **Chunks** a block of text into smaller pieces along natural boundaries such as paragraphs and sentences.
2. **Embeds** each chunk as a fixed-length `[]float32` vector, either by meaning (pretrained GloVe word vectors, so "car" lands near "automobile") or by shared words and word fragments (feature hashing, no model files needed).
3. **Stores** the results through a small `Store` interface, ready for a vector database.

It can also **serve** any embedder over an OpenAI-compatible `/v1/embeddings` HTTP endpoint that you mount in your own program.

```go
import (
	embedding "github.com/brimstone/go-embedding"
	"github.com/brimstone/go-embedding/glove" // bundled word vectors
)

chunker, _ := embedding.NewRecursiveChunker(800, 100)
embedder, _ := glove.Embedder()
store := embedding.NewMemoryStore()

p, _ := embedding.NewPipeline(chunker, embedder, embedding.PipelineOptions{Store: store})
p.Ingest(ctx, "handbook", text, map[string]string{"source": "handbook.md"})

q, _ := p.EmbedQuery(ctx, "how many vacation days do I get?")
for _, hit := range store.Search(q, 3) {
	fmt.Printf("%.3f  %s\n", hit.Score, hit.Metadata[embedding.MetaText])
}
```

## Contents

- [Installation](#installation)
- [Concepts](#concepts)
- [Chunking](#chunking)
- [Embedding](#embedding)
  - [VectorEmbedder (GloVe): matching by meaning](#vectorembedder-glove-matching-by-meaning)
  - [HashEmbedder: matching by words](#hashembedder-matching-by-words)
- [The pipeline](#the-pipeline)
- [Storing vectors](#storing-vectors)
- [Serving embeddings over HTTP](#serving-embeddings-over-http)
- [Performance](#performance)
- [Limitations](#limitations)
- [Development](#development)

## Installation

Requires Go 1.22 or newer.

```sh
go get github.com/brimstone/go-embedding
```

```go
import "github.com/brimstone/go-embedding" // package embedding
```

## Concepts

| Type | Role |
|---|---|
| `Chunker` | Splits text into `[]Chunk`. Implementations: `RecursiveChunker`, `FixedChunker`. |
| `Chunk` | A piece of the source: `Index`, `Text`, and byte offsets `Start`/`End`. |
| `Embedder` | Turns `[]string` into `[][]float32`. Implementations: `VectorEmbedder` (meaning), `HashEmbedder` (words). |
| `WordVectors` | A table of pretrained word vectors (GloVe). Bundled in the `glove` package. |
| `Pipeline` | Chunk → embed (batched, concurrent) → store. |
| `Record` | An embedded chunk: `ID`, `Vector`, `Metadata map[string]string`. |
| `Store` | Anything with `UpsertBatch([]Record) error`. `MemoryStore` is included. |
| `Handler` | An `http.Handler` that serves any `Embedder` with the OpenAI embeddings API. |

All of these are safe for concurrent use.

## Chunking

Embedding models work best on passages of a few hundred to a few thousand characters. A chunker breaks a large document into pieces of that size.

Sizes are measured in **runes** (Unicode characters), not bytes or tokens. For English text, a rough rule of thumb is 4 characters per token, so a 2,000-rune chunk is around 500 tokens.

Every chunker guarantees that:

- `source[c.Start:c.End] == c.Text`, so you can always find a chunk in the original document.
- Chunks come back in order, with leading and trailing whitespace trimmed, and are never empty.
- Every non-whitespace character of the source appears in at least one chunk; nothing is silently dropped.
- No chunk is longer than the configured size.

### RecursiveChunker (recommended)

```go
c, err := embedding.NewRecursiveChunker(800, 100) // max 800 runes, up to 100 runes of overlap
if err != nil {
	log.Fatal(err)
}
for _, ch := range c.Chunk(text) {
	fmt.Printf("#%d [%d:%d] %s\n", ch.Index, ch.Start, ch.End, ch.Text)
}
```

The chunker splits on the strongest boundary that makes pieces small enough. Pieces that are still too long are split on the next boundary down, and anything with no boundaries at all is cut hard. Neighbouring pieces are then packed together into chunks as close to the size limit as possible.

The default boundaries (`embedding.DefaultSeparators`), strongest first:

```
"\n\n"  paragraphs
"\n"    lines
". " "? " "! "   sentences
"; " ": " ", "   clauses
" "     words
```

You can pass your own boundaries, for example to split Markdown on headings first:

```go
c, err := embedding.NewRecursiveChunker(1000, 0, "\n## ", "\n### ", "\n\n", "\n", ". ", " ")
```

**Overlap** repeats the end of one chunk at the start of the next, so a sentence that straddles a boundary can still be found. With `RecursiveChunker`, overlap is made of whole pieces (for example whole sentences) totalling at most `overlap` runes. If the last piece is bigger than that, there is no overlap at that boundary.

### FixedChunker

```go
c, err := embedding.NewFixedChunker(500, 50)
```

`FixedChunker` produces windows of at most `size` runes, ignoring document structure. It still tries to break between words, and starts each overlap at a word boundary. Use it for text with no useful structure, such as transcripts or logs.

### Chunker sizes

Both constructors return `ErrInvalidChunkSize` unless `size > 0` and `0 <= overlap < size`.

## Embedding

```go
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Dimension() int
}
```

### Choosing an embedder

| | `VectorEmbedder` (GloVe) | `HashEmbedder` |
|---|---|---|
| Matches | Meaning: "car" ≈ "automobile", "doctor" ≈ "physician" | Shared words and word fragments |
| Adds to your binary | 32 MB (`glove` package) | Nothing |
| Typos, product codes, names | Unknown words are ignored | Handled well |
| Languages | English (GloVe 6B) | Any language written with spaces |

On a small benchmark of 20 queries paraphrased so that they share no content words with their target, `VectorEmbedder` ranked the right document first 80–85% of the time, against 10% for `HashEmbedder` (see [Evaluation](#evaluation)).

### VectorEmbedder (GloVe): matching by meaning

`VectorEmbedder` uses [GloVe](https://nlp.stanford.edu/projects/glove/) word vectors, learned by Stanford NLP from 6 billion words of Wikipedia and newswire text. Words used in similar contexts get similar vectors, so a query about an "automobile" finds a passage about a "car".

The vectors ship with the library in the `github.com/brimstone/go-embedding/glove` package (the 100,000 most frequent words, 300 dimensions), so there is nothing to download or convert:

```go
import (
	embedding "github.com/brimstone/go-embedding"
	"github.com/brimstone/go-embedding/glove"
)

e, err := glove.Embedder()
if err != nil {
	log.Fatal(err)
}
p, err := embedding.NewPipeline(chunker, e, embedding.PipelineOptions{Store: store})
```

`VectorEmbedder` is an ordinary `Embedder`, so it works with the pipeline, the HTTP handler, `MemoryStore` and the vector database adapter. Its dimension is `glove.Dimension` (300).

The vectors are decoded on first use, which takes about 0.1 s. Only programs that import the `glove` package carry its data, so a program using just the chunker or `HashEmbedder` stays small. Every `go get` of the module still downloads the data file (about 30 MB, once, into the module cache).

For other SIF settings, build the embedder from the vectors yourself:

```go
wv, err := glove.WordVectors()
e, err := embedding.NewVectorEmbedder(wv, embedding.SIFOptions{A: 1e-4})
```

#### Using other word vectors

To use a smaller file (the 100-dimensional vectors are a third of the size, with slightly lower quality), more words, or your own vectors in GloVe's text format, convert them with `cmd/glove-convert`, then load the result:

```sh
go run github.com/brimstone/go-embedding/cmd/glove-convert@latest \
  -in glove.6B.zip -member glove.6B.200d.txt -out glove-200d.wv -max-words 200000
```

```go
wv, err := embedding.LoadWordVectors("glove-200d.wv")
e, err := embedding.NewVectorEmbedder(wv, embedding.SIFOptions{})
```

`glove.6B.zip` (822 MB) is available from the [Stanford GloVe page](https://nlp.stanford.edu/projects/glove/) or Stanford NLP's [Hugging Face mirror](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip). The converter reads straight from the zip and:

- keeps the most frequent words (`-max-words`, default 100,000; 0 for all),
- removes the mean vector and the dominant directions, which mostly encode how common a word is rather than what it means ("All-but-the-Top", Mu & Viswanath 2018; disable with `-no-postprocess`),
- quantizes to int8, cutting the size to a quarter with a negligible effect on similarity,
- writes a checksummed binary file atomically, and prints the nearest neighbours of a few words (`-check car,king`) as a sanity check.

Conversion is deterministic: the same input produces byte-for-byte the same file on any machine. The bundled file was made this way, and the command that reproduces it is in the `glove` package documentation. To embed your own file in your binary, use `//go:embed` and `embedding.ReadWordVectors(bytes.NewReader(data))`.

#### How a text becomes a vector

The embedder uses **Smooth Inverse Frequency** (SIF; Arora, Liang & Ma, 2017), a simple method that holds its own against far more complex ones:

1. The text is split into lowercased words. Hyphenated words and contractions are looked up whole when GloVe knows them ("e-mail") and otherwise split the way GloVe's training text was ("don't" → "do" + "n't", "state-of-the-art" → its parts).
2. Each known word's vector is weighted by `a / (a + p(word))`, where `p` is how often the word occurs. Common words like "the" get almost no weight, and rare, meaningful ones dominate. GloVe files don't include frequencies, so they are estimated from each word's rank using Zipf's law.
3. The weighted vectors are averaged and normalised to unit length.

Words that aren't in the vocabulary are ignored. A text with no known words embeds to the zero vector.

`SIFOptions.A` sets `a` (default `1e-3`; useful range `1e-4`–`1e-2`). Lower values suppress common words more strongly.

#### Optional: remove the common component

The second half of SIF removes the one direction that all of your texts share. Fit it on a sample of the texts you are indexing:

```go
var sample []string
for _, ch := range chunker.Chunk(allMyDocuments) {
	sample = append(sample, ch.Text)
}
e = e.WithCommonComponent(sample) // returns a new embedder; e is unchanged
```

As with IDF, use the same fitted embedder for indexing and queries, and re-embed stored documents if you refit it.

#### Exploring the vectors

`WordVectors` can answer questions about single words, which helps when checking why a search did or didn't match:

```go
s, ok := wv.Similarity("car", "automobile")  // 0.56, true
nb, ok := wv.MostSimilar("doctor", 5)        // physician 0.68, doctors 0.61, pharmacist 0.58, ...
v, ok := wv.Vector("car")                    // []float32 copy
```

#### Evaluation

`glove_eval_test.go` contains a small retrieval benchmark: 20 short documents on different topics and 20 queries, each paraphrasing one document without using its content words. For example, "how often should I service my automobile motor lubricant" should find "Changing the engine oil every five thousand miles keeps your car running smoothly." Results with the default settings unless noted:

| Embedder | Right document 1st | In top 3 | MRR |
|---|---|---|---|
| `HashEmbedder` | 10% | 25% | 0.25 |
| `glove.Embedder()` | 80% | 100% | 0.90 |
| `glove.Embedder()` + `WithCommonComponent` | 85% | 100% | 0.93 |
| 100d vectors, converted yourself | 80% | 100% | 0.88 |
| 300d vectors converted with `-no-postprocess` | 70% | 95% | 0.84 |

With only 20 queries, a difference of 5 points is a single query, so read the small gaps as noise. The evaluation runs as part of `go test` on the bundled vectors. To include your own converted files:

```sh
EMBEDDING_GLOVE=my-vectors.wv go test -run GloVe -v .
```

The best evaluation is your own: a few dozen real queries from your users, each paired with the passage it should find.

#### License

The GloVe 6B vectors are released by Stanford NLP under the [Public Domain Dedication and License (PDDL)](https://opendatacommons.org/licenses/pddl/1-0/), so the bundled file (and any you convert) can be redistributed freely. Citing Pennington, Socher & Manning, "GloVe: Global Vectors for Word Representation" (EMNLP 2014) is appreciated. Other GloVe releases and other vector sets have their own terms.

### HashEmbedder: matching by words

`HashEmbedder` embeds text with **feature hashing**. It needs no model files and no network access, and the same text always gives the same vector.

```go
e, err := embedding.NewHashEmbedder(embedding.DefaultHashOptions())
vecs, err := e.Embed(ctx, []string{"first text", "second text"})
```

Each text is split into lowercased words. The embedder then hashes three kinds of feature into the vector, each with a hash-chosen sign:

- **words**: `database`
- **word pairs**, for some sense of word order: `vector database`
- **character n-grams** of each word: `<da`, `dat`, `ata`, …. These make the vector tolerant of typos and word endings, so `databse` lands near `database`, and `store` near `stores`.

The result is normalised to unit length, so cosine similarity and dot product give the same ranking. Text with no letters or digits embeds to the zero vector.

#### Options

```go
embedding.HashOptions{
	Dimension:   512,  // vector length; more dimensions = fewer hash collisions
	MinN:        3,    // character n-gram range; MinN = 0 disables n-grams
	MaxN:        5,
	WordBigrams: true, // include adjacent word pairs
}
```

`DefaultHashOptions()` returns the values shown above. Some guidance:

- **Dimension.** 256–1024 is a sensible range. Higher dimensions are more accurate for long chunks but slower and use more storage.
- **N-grams off** (`HashOptions{Dimension: 512}`). About twice as fast, but only exact word matches count.

#### IDF weighting

Without weighting, common words like "the" and "and" count as much as rare, meaningful ones. `WithIDF` learns which features are common across a set of texts and turns them down:

```go
base, _ := embedding.NewHashEmbedder(embedding.DefaultHashOptions())

var corpus []string
for _, ch := range chunker.Chunk(allMyDocuments) {
	corpus = append(corpus, ch.Text)
}
e := base.WithIDF(corpus) // returns a new embedder; base is unchanged
```

For good results, use the same IDF-weighted embedder for indexing and for queries. If you rebuild it with a different corpus, re-embed your stored documents.

### Bring your own model

`VectorEmbedder` understands words but not sentences: it ignores word order and handles negation poorly. For more, implement `Embedder` around a full sentence-embedding model, for example one served by another process. Everything else in the library (pipeline, handler, stores) works unchanged.

```go
type MyModel struct{ /* ... */ }

func (m *MyModel) Embed(ctx context.Context, texts []string) ([][]float32, error) { /* ... */ }
func (m *MyModel) Dimension() int { return 768 }
```

`Dimension` may return 0 if the size isn't known until the first call.

### Cosine similarity

`embedding.Cosine(a, b)` returns the cosine similarity of two vectors, in [-1, 1].

## The pipeline

`Pipeline` ties a chunker, an embedder and an optional store together:

```go
p, err := embedding.NewPipeline(chunker, embedder, embedding.PipelineOptions{
	Store:     store, // optional; nil just returns the records
	BatchSize: 32,    // chunks per Embed call (default 32)
	Workers:   8,     // concurrent Embed calls (default GOMAXPROCS)
})

records, err := p.Ingest(ctx, "doc-42", text, map[string]string{
	"source": "contracts/acme.pdf",
	"lang":   "en",
})
```

`Ingest`:

1. chunks `text`,
2. embeds the chunks in batches across `Workers` goroutines,
3. builds one `Record` per chunk, in chunk order,
4. writes all of the document's records to the store in **one** `UpsertBatch` call, so if anything fails, nothing is stored,
5. returns the records.

If any batch fails or `ctx` is cancelled, the other workers stop and the first error is returned. Empty or whitespace-only text returns no records and no error.

### Records

Each record looks like this:

```go
embedding.Record{
	ID:     "doc-42#3",            // "<docID>#<chunk index>"
	Vector: []float32{...},
	Metadata: map[string]string{
		"source":   "contracts/acme.pdf", // your metadata, copied in
		"lang":     "en",
		"doc_id":   "doc-42",             // embedding.MetaDocID
		"chunk":    "3",                  // embedding.MetaChunk
		"start":    "2412",               // embedding.MetaStart (byte offset)
		"end":      "3180",               // embedding.MetaEnd
		"text":     "…the chunk text…",   // embedding.MetaText
	},
}
```

If your metadata uses one of the reserved keys, the pipeline's value wins. Re-ingesting a document with the same ID overwrites chunks `#0…#n`. If the new version has fewer chunks, the old trailing chunks remain; delete them from your store if that matters.

### Queries

Embed search queries with the same embedder used for indexing:

```go
q, err := p.EmbedQuery(ctx, "termination clause notice period")
```

## Storing vectors

`Store` is deliberately tiny:

```go
type Store interface {
	UpsertBatch(recs []Record) error
}
```

### MemoryStore

`MemoryStore` keeps records in a map and searches them exactly by cosine similarity. It's handy for tests, demos and small collections (a few thousand chunks):

```go
store := embedding.NewMemoryStore()
// ... ingest ...
hits := store.Search(queryVec, 5) // []Hit, best first; Hit embeds Record and adds Score
rec, ok := store.Get("doc-42#3")
n := store.Len()
```

### Using a vector database

`Record` has the same shape as a typical vector database record, so an adapter takes only a few lines. For example, for the companion `vector` package:

```go
type vectorStore struct{ c *vector.Collection }

func (s vectorStore) UpsertBatch(recs []embedding.Record) error {
	out := make([]vector.Record, len(recs))
	for i, r := range recs {
		out[i] = vector.Record{ID: r.ID, Vector: r.Vector, Content: r.Metadata[embedding.MetaText], Metadata: r.Metadata}
	}
	return s.c.UpsertBatch(out)
}

coll, err := db.GetOrCreateCollection("docs", vector.Options{
	Dimension: embedder.Dimension(),
	Metric:    vector.Cosine,
})
p, err := embedding.NewPipeline(chunker, embedder, embedding.PipelineOptions{Store: vectorStore{coll}})
```

The chunk text goes into the record's `Content` (and the `text` metadata key), so search results can be shown or passed to an LLM without going back to the original documents. Because `VectorEmbedder` and `HashEmbedder` also satisfy `vector.Embedder`, it can be passed straight to `Collection.SearchText` and `Collection.UpsertDocuments`.

When re-ingesting a document, first remove its old chunks with `coll.DeleteWhere(vector.Where(embedding.MetaDocID, docID))`.

## Serving embeddings over HTTP

`Handler` is an `http.Handler` that speaks the OpenAI embeddings API. Mount it in your own server and any OpenAI-compatible client library or tool can use your embedders:

```go
e, _ := embedding.NewHashEmbedder(embedding.DefaultHashOptions())

mux := http.NewServeMux()
mux.Handle("POST /v1/embeddings", &embedding.Handler{
	Models: map[string]embedding.Embedder{
		"hash-512": e,
	},
	MaxInputs:    2048,    // texts per request (default 2048)
	MaxBodyBytes: 8 << 20, // request body limit (default 8 MiB)
})
log.Fatal(http.ListenAndServe(":8080", mux))
```

Point a client at base URL `http://localhost:8080/v1` with model `hash-512`:

```sh
curl -s localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "hash-512", "input": ["first text", "second text"]}'
```

```json
{
  "object": "list",
  "model": "hash-512",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, ...]},
    {"object": "embedding", "index": 1, "embedding": [...]}
  ],
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

### Request fields

| Field | Notes |
|---|---|
| `model` | Must be a key of `Models`. Optional when exactly one model is configured. |
| `input` | A string or an array of strings. Arrays of token IDs are not supported. |
| `encoding_format` | `"float"` (default) or `"base64"`: little-endian float32s, as OpenAI uses. |
| `dimensions` | Optional. If given, it must equal the model's dimension; vectors are never truncated. |

### Behaviour

- **Errors** use OpenAI's format, `{"error": {"message", "type", "param", "code"}}`, with these statuses:
  - 400: bad request
  - 404 with code `model_not_found`: unknown model
  - 405: anything but `POST`
  - 413: body too large
  - 500: the embedder failed
- **Usage** counts words, as an approximation of tokens.
- The handler is not tied to a path; mount it wherever you like.
- **Authentication, rate limiting and CORS** are not included. Wrap the handler with your own middleware.

## Performance

Measured on an Intel i9-13900HK (20 threads) with about 1 MiB of English prose (`go test -bench .`):

| Operation | Throughput |
|---|---|
| `RecursiveChunker`, 1024-rune chunks | ~520 MB/s |
| `FixedChunker`, 1024-rune chunks | ~580 MB/s |
| `HashEmbedder`, default options | ~35 MB/s per core |
| `HashEmbedder`, words only | ~76 MB/s per core |
| `HashEmbedder`, one short query | ~4.5 µs |
| `Pipeline.Ingest`, all cores | ~33 MB/s |
| `MemoryStore.Search`, 1,800 records | ~1.1 ms |
| `VectorEmbedder`, 300d GloVe | ~31 MB/s per core |
| `VectorEmbedder`, one short query | ~2.6 µs |
| `LoadWordVectors` or first `glove` use, 100k × 300 (32 MB) | ~90 ms |
| `WordVectors.MostSimilar`, 100k words | ~13 ms |
| `glove-convert`, 300d, 100k words | ~8 s |

For `HashEmbedder`, embedding cost grows with the number of n-grams per word. For faster indexing, narrow the n-gram range or turn n-grams off. A loaded 100k × 300 `WordVectors` uses about 40 MB of memory.

## Limitations

- **`HashEmbedder` is lexical.** It finds text that shares words and word fragments with the query, and doesn't understand synonyms or paraphrases. Use `VectorEmbedder` for that.
- **`VectorEmbedder` is a bag of words.** Word order is lost ("dog bites man" = "man bites dog") and negation is weak ("not good" lands near "good"). Words missing from the vocabulary, such as product codes, rare names, typos and terms newer than 2014, are ignored. GloVe 6B is English only.
- **Hash collisions.** Different features can land in the same slot. Larger `Dimension` values reduce this.
- **Character-based sizes.** Chunk sizes are in runes, not model tokens. If your model has a hard token limit, leave headroom.
- **Word splitting.** Words are runs of Unicode letters, digits and combining marks. Scripts written without spaces (Chinese, Japanese, Thai) become very long "words". Character n-grams still give useful matches for them, but word features are weak.
- **`MemoryStore` is brute force.** Use a real vector index for large collections.

## Development

```sh
go fmt ./...
go vet ./...
go test ./...                               # unit tests and examples
go test -race ./...                         # race detector (needs cgo)
go test -run '^$' -bench . -benchmem        # benchmarks
go test -run '^$' -fuzz FuzzChunkers -fuzztime 60s          # fuzz the chunkers
go test -run '^$' -fuzz FuzzReadWordVectors -fuzztime 60s   # fuzz the word-vector file reader
go test -run GloVe -v .                                     # retrieval evaluation on the bundled vectors
```

### Layout

| File | Contents |
|---|---|
| `chunk.go` | `Chunk`, `Chunker`, `FixedChunker`, `RecursiveChunker` |
| `embed.go` | `Embedder`, `HashEmbedder`, `Cosine` |
| `wordvec.go` | `WordVectors`: GloVe parsing, post-processing, quantization, file format |
| `sif.go` | `VectorEmbedder` |
| `glove/` | Bundled GloVe vectors (a `.wv` file embedded with `//go:embed`) |
| `cmd/glove-convert` | Converts GloVe text files or zips into `.wv` files |
| `glove_eval_test.go` | Word-similarity and retrieval evaluation |
| `pipeline.go` | `Pipeline`, `Record`, `Store`, metadata keys |
| `memstore.go` | `MemoryStore` |
| `handler.go` | `Handler`, the OpenAI-compatible endpoint |
| `example_test.go` | Runnable examples (shown by `go doc`) |

Contributions must keep the library dependency-free: standard library only.
