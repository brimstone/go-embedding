package embedding

import (
	"context"
	"errors"
	"fmt"
	"math"
	"unicode"
	"unicode/utf8"
)

// Embedder converts texts into vectors. Implementations must be safe for
// concurrent use.
type Embedder interface {
	// Embed returns one vector per input text, in input order.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Dimension is the length of the returned vectors, or 0 if it is not
	// yet known.
	Dimension() int
}

// HashOptions configures a HashEmbedder.
type HashOptions struct {
	// Dimension is the number of hash buckets, i.e. the vector length.
	Dimension int
	// MinN and MaxN bound the character n-grams taken from each word
	// (wrapped as "<word>"). Character n-grams make the embedding robust to
	// typos and inflections. MinN == 0 disables them.
	MinN, MaxN int
	// WordBigrams adds features for adjacent word pairs, which captures
	// some word order.
	WordBigrams bool
}

// DefaultHashOptions returns the options used by NewHashEmbedder when none
// are customised: 512 dimensions, character 3- to 5-grams and word bigrams.
func DefaultHashOptions() HashOptions {
	return HashOptions{Dimension: 512, MinN: 3, MaxN: 5, WordBigrams: true}
}

// Relative feature weights. Each of a word's k character n-grams gets
// charWeight/sqrt(k), so every word's n-grams contribute the same L2 energy
// regardless of word length (short words would otherwise dominate).
const (
	wordWeight   = 1.0
	bigramWeight = 0.5
	charWeight   = 1.0
)

// Hash seeds, so the same bytes hash differently per feature type.
const (
	seedWord   = 0x9e3779b97f4a7c15
	seedBigram = 0xc2b2ae3d27d4eb4f
	seedChar   = 0x165667b19e3779f9
)

// HashEmbedder embeds text by feature hashing: each lowercased word, word
// bigram and character n-gram is hashed to a bucket and a sign, the signed
// weights are summed, and the result is L2 normalised. It is deterministic,
// needs no model or network access, and is fast, but only captures lexical
// (not semantic) similarity. Use WithIDF to down-weight common words.
//
// A HashEmbedder is immutable and safe for concurrent use.
type HashEmbedder struct {
	opts HashOptions
	idf  []float32 // per-bucket weight, nil for none
}

// NewHashEmbedder returns a HashEmbedder with the given options.
func NewHashEmbedder(opts HashOptions) (*HashEmbedder, error) {
	if opts.Dimension <= 0 {
		return nil, fmt.Errorf("embedding: dimension %d must be positive", opts.Dimension)
	}
	if opts.MinN < 0 || (opts.MinN > 0 && opts.MaxN < opts.MinN) {
		return nil, fmt.Errorf("embedding: invalid n-gram range [%d, %d]", opts.MinN, opts.MaxN)
	}
	return &HashEmbedder{opts: opts}, nil
}

// Dimension implements Embedder.
func (e *HashEmbedder) Dimension() int { return e.opts.Dimension }

// WithIDF returns a copy of e that weights each bucket by its inverse
// document frequency over corpus, so features that occur in most documents
// (like "the") contribute little. corpus should be representative of the
// texts being embedded, e.g. the chunks of the documents being indexed.
func (e *HashEmbedder) WithIDF(corpus []string) *HashEmbedder {
	dim := e.opts.Dimension
	df := make([]int, dim)
	v := make([]float32, dim)
	var s scratch
	for _, doc := range corpus {
		clear(v)
		e.accumulate(v, doc, &s)
		for b, x := range v {
			if x != 0 {
				df[b]++
			}
		}
	}
	n := float64(len(corpus))
	idf := make([]float32, dim)
	for b := range idf {
		idf[b] = float32(math.Log((1+n)/(1+float64(df[b]))) + 1)
	}
	return &HashEmbedder{opts: e.opts, idf: idf}
}

// Embed implements Embedder. Texts with no words embed to the zero vector.
func (e *HashEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	dim := e.opts.Dimension
	out := make([][]float32, len(texts))
	backing := make([]float32, len(texts)*dim)
	var s scratch
	for i, t := range texts {
		if i%64 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		v := backing[i*dim : (i+1)*dim : (i+1)*dim]
		e.accumulate(v, t, &s)
		if e.idf != nil {
			for b := range v {
				v[b] *= e.idf[b]
			}
		}
		normalize(v)
		out[i] = v
	}
	return out, nil
}

// scratch holds reusable buffers for one goroutine.
type scratch struct {
	buf  []byte // "<word>", lowercased
	offs []int  // byte offset of each rune in buf, plus len(buf)
}

func isWordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.Is(unicode.Mn, r)
}

// accumulate adds the hashed features of text to v.
func (e *HashEmbedder) accumulate(v []float32, text string, s *scratch) {
	var prev uint64
	havePrev := false
	for i := 0; i < len(text); {
		r, w := utf8.DecodeRuneInString(text[i:])
		if !isWordRune(r) {
			i += w
			continue
		}
		s.buf = append(s.buf[:0], '<')
		s.offs = append(s.offs[:0], 0)
		for i < len(text) {
			r, w = utf8.DecodeRuneInString(text[i:])
			if !isWordRune(r) {
				break
			}
			s.offs = append(s.offs, len(s.buf))
			s.buf = utf8.AppendRune(s.buf, unicode.ToLower(r))
			i += w
		}
		s.offs = append(s.offs, len(s.buf))
		s.buf = append(s.buf, '>')
		s.offs = append(s.offs, len(s.buf))

		hw := hashBytes(seedWord, s.buf[1:len(s.buf)-1])
		e.add(v, hw, wordWeight)
		if e.opts.WordBigrams && havePrev {
			e.add(v, mix(prev*0x100000001b3^hw^seedBigram), bigramWeight)
		}
		prev, havePrev = hw, true

		if e.opts.MinN == 0 {
			continue
		}
		runes := len(s.offs) - 1
		count := 0
		for n := e.opts.MinN; n <= e.opts.MaxN && n <= runes; n++ {
			count += runes - n + 1
		}
		if count == 0 {
			continue
		}
		wt := float32(charWeight / math.Sqrt(float64(count)))
		for n := e.opts.MinN; n <= e.opts.MaxN && n <= runes; n++ {
			for j := 0; j+n <= runes; j++ {
				e.add(v, hashBytes(seedChar, s.buf[s.offs[j]:s.offs[j+n]]), wt)
			}
		}
	}
}

// add adds ±w to the bucket selected by h.
func (e *HashEmbedder) add(v []float32, h uint64, w float32) {
	b := h % uint64(len(v))
	if h>>63 != 0 {
		w = -w
	}
	v[b] += w
}

// hashBytes is 64-bit FNV-1a seeded with seed, followed by a finaliser so
// the low bits used for bucketing are well mixed.
func hashBytes(seed uint64, p []byte) uint64 {
	h := uint64(0xcbf29ce484222325) ^ seed
	for _, c := range p {
		h ^= uint64(c)
		h *= 0x100000001b3
	}
	return mix(h)
}

// mix is the splitmix64 finaliser.
func mix(h uint64) uint64 {
	h ^= h >> 30
	h *= 0xbf58476d1ce4e5b9
	h ^= h >> 27
	h *= 0x94d049bb133111eb
	h ^= h >> 31
	return h
}

// normalize scales v to unit L2 length. The zero vector is left unchanged.
func normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

// Cosine returns the cosine similarity of a and b, in [-1, 1]. It returns
// 0 if either vector is zero and panics if their lengths differ.
func Cosine(a, b []float32) float32 {
	if len(a) != len(b) {
		panic(errors.New("embedding: vector length mismatch"))
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / math.Sqrt(na*nb))
}
