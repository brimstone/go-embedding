package embedding

import (
	"bytes"
	"context"
	"errors"
	"math"
	"unicode"
	"unicode/utf8"
)

// SIFOptions configures a VectorEmbedder.
type SIFOptions struct {
	// A is the SIF smoothing parameter. Each word is weighted by
	// A / (A + p(word)), where p is the word's estimated probability, so
	// very common words ("the", "of") contribute little and rare,
	// meaningful words dominate. Useful values are 1e-4 to 1e-2.
	// Default 1e-3.
	A float64
}

// VectorEmbedder embeds text by averaging pretrained word vectors (such as
// GloVe) using Smooth Inverse Frequency weighting (Arora, Liang & Ma,
// "A Simple but Tough-to-Beat Baseline for Sentence Embeddings", 2017).
//
// Unlike HashEmbedder it captures meaning learned from a large corpus:
// "car" and "automobile" get similar vectors even though they share no
// letters. Like any bag-of-words model it ignores word order and handles
// negation poorly. Words not in the vocabulary are ignored, and a text with
// no known words embeds to the zero vector.
//
// Call WithCommonComponent with a sample of your texts to also remove the
// direction shared by all of them, which completes the SIF method and
// usually improves retrieval.
//
// A VectorEmbedder is immutable and safe for concurrent use.
type VectorEmbedder struct {
	wv     *WordVectors
	weight []float32 // SIF weight × quantisation scale, per word
	pc     []float32 // common component to remove, or nil
}

// NewVectorEmbedder returns a VectorEmbedder using wv.
func NewVectorEmbedder(wv *WordVectors, opts SIFOptions) (*VectorEmbedder, error) {
	if wv == nil {
		return nil, errors.New("embedding: nil word vectors")
	}
	a := opts.A
	if a == 0 {
		a = 1e-3
	}
	if !(a > 0) || math.IsInf(a, 0) {
		return nil, errors.New("embedding: SIF parameter A must be positive")
	}
	w := make([]float32, wv.Len())
	for i := range w {
		w[i] = float32(a/(a+float64(wv.prob[i]))) * wv.scale[i]
	}
	return &VectorEmbedder{wv: wv, weight: w}, nil
}

// Dimension implements Embedder.
func (e *VectorEmbedder) Dimension() int { return e.wv.dim }

// WordVectors returns the underlying word vectors.
func (e *VectorEmbedder) WordVectors() *WordVectors { return e.wv }

// WithCommonComponent returns a copy of e that removes the first principal
// component of corpus's embeddings from every embedding, as in the SIF
// paper. corpus should be representative of the texts being indexed, for
// example a few thousand of their chunks. Embeddings from the returned
// embedder are not comparable with those from e.
func (e *VectorEmbedder) WithCommonComponent(corpus []string) *VectorEmbedder {
	dim := e.wv.dim
	var x []float32
	var buf []byte
	v := make([]float32, dim)
	for _, t := range corpus {
		clear(v)
		if e.accumulate(v, t, &buf) > 0 {
			x = append(x, v...)
		}
	}
	out := &VectorEmbedder{wv: e.wv, weight: e.weight}
	if len(x) == 0 {
		return out
	}
	pcs := topComponents(covariance(x, dim), dim, 1)
	if len(pcs) == 1 {
		out.pc = make([]float32, dim)
		for j, u := range pcs[0] {
			out.pc[j] = float32(u)
		}
	}
	return out
}

// Embed implements Embedder.
func (e *VectorEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	dim := e.wv.dim
	out := make([][]float32, len(texts))
	backing := make([]float32, len(texts)*dim)
	var buf []byte
	for i, t := range texts {
		if i%64 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		v := backing[i*dim : (i+1)*dim : (i+1)*dim]
		if e.accumulate(v, t, &buf) > 0 {
			if e.pc != nil {
				var p float32
				for j, u := range e.pc {
					p += v[j] * u
				}
				for j, u := range e.pc {
					v[j] -= p * u
				}
			}
			normalize(v)
		}
		out[i] = v
	}
	return out, nil
}

// accumulate sets v to the SIF-weighted average of text's word vectors and
// returns the number of known words.
func (e *VectorEmbedder) accumulate(v []float32, text string, buf *[]byte) int {
	n := 0
	dim := e.wv.dim
	add := func(i int32) {
		w := e.weight[i]
		row := e.wv.data[int(i)*dim : int(i+1)*dim]
		for j, q := range row {
			v[j] += w * float32(q)
		}
		n++
	}
	*buf = forEachWord(text, *buf, func(word []byte) { e.lookup(word, add) })
	if n > 1 {
		inv := 1 / float32(n)
		for j := range v {
			v[j] *= inv
		}
	}
	return n
}

// lookup calls add for the vocabulary entries matching word. A word that is
// not in the vocabulary is split on hyphens, and then contractions are
// split the way GloVe's corpus was tokenised: "don't" → "do" + "n't",
// "it's" → "it" + "'s".
func (e *VectorEmbedder) lookup(word []byte, add func(int32)) {
	if i, ok := e.wv.index[string(word)]; ok {
		add(i)
		return
	}
	for _, part := range bytes.Split(word, []byte("-")) {
		if len(part) == 0 {
			continue
		}
		if i, ok := e.wv.index[string(part)]; ok {
			add(i)
			continue
		}
		j := bytes.IndexByte(part, '\'')
		if j < 0 {
			continue
		}
		head, tail := part[:j], part[j:]
		if bytes.HasSuffix(part, []byte("n't")) && len(part) > 3 {
			head, tail = part[:len(part)-3], part[len(part)-3:]
		}
		for _, p := range [][]byte{head, tail} {
			if i, ok := e.wv.index[string(p)]; ok {
				add(i)
			}
		}
	}
}

// forEachWord calls fn with each lowercased word of text, reusing buf.
// Words are runs of letters, digits and combining marks. An apostrophe or
// hyphen between two word characters is kept, so "e-mail" and "don't" stay
// whole; typographic apostrophes are normalised to '.
func forEachWord(text string, buf []byte, fn func(word []byte)) []byte {
	for i := 0; i < len(text); {
		r, w := utf8.DecodeRuneInString(text[i:])
		if !isWordRune(r) {
			i += w
			continue
		}
		buf = buf[:0]
		for i < len(text) {
			r, w = utf8.DecodeRuneInString(text[i:])
			if isWordRune(r) {
				buf = utf8.AppendRune(buf, unicode.ToLower(r))
				i += w
				continue
			}
			if r == '\'' || r == '’' || r == '-' {
				if next, _ := utf8.DecodeRuneInString(text[i+w:]); i+w < len(text) && isWordRune(next) {
					if r == '-' {
						buf = append(buf, '-')
					} else {
						buf = append(buf, '\'')
					}
					i += w
					continue
				}
			}
			break
		}
		fn(buf)
	}
	return buf
}
