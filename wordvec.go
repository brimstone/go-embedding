package embedding

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"unicode/utf8"
)

// WordVectors is an immutable table of pretrained word vectors, such as
// GloVe, together with an estimate of each word's frequency. Vectors are
// stored quantized to int8 with one scale per word, which takes a quarter
// of the memory of float32 and changes cosine similarities by well under
// 1%.
//
// Load a table converted ahead of time with LoadWordVectors or
// ReadWordVectors (see cmd/glove-convert), or build one from GloVe's text
// format with ReadGloVe. Use it with NewVectorEmbedder, or directly with
// Similarity and MostSimilar.
//
// A WordVectors is safe for concurrent use.
type WordVectors struct {
	dim   int
	words []string
	index map[string]int32
	prob  []float32 // estimated unigram probability
	scale []float32 // vector i is scale[i] * data[i*dim:(i+1)*dim]
	norm  []float32 // L2 norm of vector i
	data  []int8
}

// ErrCorruptWordVectors is returned (wrapped) when a word vector file is
// truncated, fails its checksum or is otherwise invalid.
var ErrCorruptWordVectors = errors.New("embedding: corrupt word vectors")

// GloVeOptions configures ReadGloVe.
type GloVeOptions struct {
	// MaxWords keeps only the first MaxWords words. GloVe files are sorted
	// by descending frequency, so this keeps the most common words.
	// Zero keeps every word.
	MaxWords int
	// Components is the number of dominant principal components removed
	// (after subtracting the mean vector), following "All-but-the-Top"
	// (Mu & Viswanath, 2018). These directions mostly encode word
	// frequency rather than meaning, and removing them improves
	// similarity. Zero selects Dimension/100 (at least 1).
	Components int
	// NoPostProcessing disables mean and component removal.
	NoPostProcessing bool
}

// ReadGloVe reads word vectors in GloVe's text format: one word per line
// followed by its vector components, separated by spaces, most frequent
// word first. Word frequencies, which GloVe files do not include, are
// estimated from each word's rank using Zipf's law.
//
// Lines with more fields than the dimension (words containing spaces, as
// in some GloVe releases) are accepted; repeated words keep their first
// vector.
func ReadGloVe(r io.Reader, opts GloVeOptions) (*WordVectors, error) {
	if opts.MaxWords < 0 || opts.Components < 0 {
		return nil, errors.New("embedding: negative GloVeOptions value")
	}
	br := bufio.NewReaderSize(r, 1<<20)
	var (
		dim   int
		words []string
		seen  = map[string]bool{}
		data  []float32
		total int // lines read, for the Zipf normalisation
	)
	for lineNo := 1; ; lineNo++ {
		line, err := br.ReadSlice('\n')
		if errors.Is(err, bufio.ErrBufferFull) {
			// Very long line: fall back to an allocating read.
			rest, err2 := br.ReadBytes('\n')
			line = append(append([]byte(nil), line...), rest...)
			err = err2
		}
		if len(bytes.TrimSpace(line)) > 0 {
			total++
			if opts.MaxWords == 0 || len(words) < opts.MaxWords {
				fields := bytes.Fields(line)
				if dim == 0 {
					dim = len(fields) - 1
					if dim < 1 || dim > 1<<16 {
						return nil, fmt.Errorf("embedding: glove line %d: bad dimension %d", lineNo, dim)
					}
				}
				if len(fields) < dim+1 {
					return nil, fmt.Errorf("embedding: glove line %d: %d values, want %d", lineNo, len(fields)-1, dim)
				}
				word := string(bytes.Join(fields[:len(fields)-dim], []byte(" ")))
				if !seen[word] {
					seen[word] = true
					for _, f := range fields[len(fields)-dim:] {
						x, perr := strconv.ParseFloat(string(f), 32)
						if perr != nil || math.IsNaN(x) || math.IsInf(x, 0) {
							return nil, fmt.Errorf("embedding: glove line %d: bad value %q", lineNo, f)
						}
						data = append(data, float32(x))
					}
					words = append(words, word)
				}
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
	}
	if len(words) == 0 {
		return nil, errors.New("embedding: glove input contains no vectors")
	}

	// Zipf-Mandelbrot estimate of unigram probability from rank r (1-based):
	// p(r) ∝ 1/(r + 2.7).
	var z float64
	for r := 1; r <= total; r++ {
		z += 1 / (float64(r) + 2.7)
	}
	prob := make([]float32, len(words))
	for i := range prob {
		prob[i] = float32(1 / ((float64(i+1) + 2.7) * z))
	}

	if !opts.NoPostProcessing {
		k := opts.Components
		if k == 0 {
			k = max(1, dim/100)
		}
		allButTheTop(data, dim, min(k, dim, len(words)-1))
	}
	return quantize(dim, words, prob, data), nil
}

// allButTheTop subtracts the mean vector from every row of x and then
// removes the projection onto the top k principal components.
func allButTheTop(x []float32, dim, k int) {
	n := len(x) / dim
	mean := make([]float64, dim)
	for i := 0; i < n; i++ {
		for j, v := range x[i*dim : (i+1)*dim] {
			mean[j] += float64(v)
		}
	}
	for j := range mean {
		mean[j] /= float64(n)
	}
	for i := 0; i < n; i++ {
		row := x[i*dim : (i+1)*dim]
		for j := range row {
			row[j] -= float32(mean[j])
		}
	}
	if k <= 0 {
		return
	}
	pcs := topComponents(covariance(x, dim), dim, k)
	for i := 0; i < n; i++ {
		row := x[i*dim : (i+1)*dim]
		for _, u := range pcs {
			var p float64
			for j, v := range row {
				p += float64(float64(v) * u[j])
			}
			for j := range row {
				row[j] -= float32(p * u[j])
			}
		}
	}
}

// covarianceChunks is the fixed number of contiguous row ranges that
// covariance sums separately. Fixing it (rather than using GOMAXPROCS) makes
// the floating-point summation order, and so the result, identical on every
// machine.
const covarianceChunks = 16

// covariance returns XᵀX (dim×dim, row-major) for the rows of x, computed
// in parallel. The result is deterministic.
//
// Throughout this file, products are wrapped in explicit float64
// conversions, which the Go spec guarantees prevents the compiler from
// fusing them into FMA instructions; this keeps converted files
// byte-for-byte identical across CPU architectures.
func covariance(x []float32, dim int) []float64 {
	n := len(x) / dim
	chunks := min(covarianceChunks, max(1, n))
	parts := make([][]float64, chunks)
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	var wg sync.WaitGroup
	for w := range chunks {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer func() { <-sem; wg.Done() }()
			c := make([]float64, dim*dim)
			row := make([]float64, dim)
			for i := w * n / chunks; i < (w+1)*n/chunks; i++ {
				for j, v := range x[i*dim : (i+1)*dim] {
					row[j] = float64(v)
				}
				for a, va := range row {
					ca := c[a*dim : a*dim+dim]
					for b := a; b < dim; b++ {
						ca[b] += float64(va * row[b])
					}
				}
			}
			parts[w] = c
		}()
	}
	wg.Wait()
	c := parts[0]
	for _, p := range parts[1:] {
		for i, v := range p {
			c[i] += v
		}
	}
	for a := 0; a < dim; a++ { // mirror the upper triangle
		for b := 0; b < a; b++ {
			c[a*dim+b] = c[b*dim+a]
		}
	}
	return c
}

// topComponents returns the k leading unit eigenvectors of the symmetric
// matrix c by power iteration with deflation. c is modified.
func topComponents(c []float64, dim, k int) [][]float64 {
	var out [][]float64
	next := make([]float64, dim)
	for range k {
		v := make([]float64, dim)
		for j := range v {
			v[j] = 1 / math.Sqrt(float64(dim)) * (1 + 0.01*float64(j%7)) // deterministic start
		}
		var lambda float64
		for iter := 0; iter < 1000; iter++ {
			for a := 0; a < dim; a++ {
				var s float64
				for b, cv := range c[a*dim : a*dim+dim] {
					s += float64(cv * v[b])
				}
				next[a] = s
			}
			var nn, delta float64
			for _, s := range next {
				nn += float64(s * s)
			}
			nn = math.Sqrt(nn)
			if nn == 0 {
				return out
			}
			for j := range v {
				nv := next[j] / nn
				delta += float64((nv - v[j]) * (nv - v[j]))
				v[j] = nv
			}
			lambda = nn
			if delta < 1e-20 {
				break
			}
		}
		out = append(out, v)
		for a := 0; a < dim; a++ { // deflate: c -= λ v vᵀ
			for b := 0; b < dim; b++ {
				c[a*dim+b] -= float64(lambda * v[a] * v[b])
			}
		}
	}
	return out
}

// quantize builds a WordVectors from float32 rows.
func quantize(dim int, words []string, prob []float32, x []float32) *WordVectors {
	n := len(words)
	wv := &WordVectors{
		dim:   dim,
		words: words,
		prob:  prob,
		scale: make([]float32, n),
		data:  make([]int8, n*dim),
	}
	for i := 0; i < n; i++ {
		row := x[i*dim : (i+1)*dim]
		var m float32
		for _, v := range row {
			m = max(m, float32(math.Abs(float64(v))))
		}
		if m == 0 {
			continue
		}
		s := m / 127
		wv.scale[i] = s
		q := wv.data[i*dim : (i+1)*dim]
		for j, v := range row {
			q[j] = int8(math.Round(float64(v / s)))
		}
	}
	wv.finish()
	return wv
}

// finish builds the index and norms.
func (wv *WordVectors) finish() {
	wv.index = make(map[string]int32, len(wv.words))
	wv.norm = make([]float32, len(wv.words))
	for i, w := range wv.words {
		wv.index[w] = int32(i)
		var s int64
		for _, q := range wv.row(i) {
			s += int64(q) * int64(q)
		}
		wv.norm[i] = wv.scale[i] * float32(math.Sqrt(float64(s)))
	}
}

func (wv *WordVectors) row(i int) []int8 { return wv.data[i*wv.dim : (i+1)*wv.dim] }

// Dim returns the vector dimension.
func (wv *WordVectors) Dim() int { return wv.dim }

// Len returns the number of words.
func (wv *WordVectors) Len() int { return len(wv.words) }

// Word returns the i-th word, in descending order of frequency.
func (wv *WordVectors) Word(i int) string { return wv.words[i] }

// Vector returns a copy of word's vector. Lookups are exact: GloVe's
// common releases contain only lowercase words.
func (wv *WordVectors) Vector(word string) ([]float32, bool) {
	i, ok := wv.index[word]
	if !ok {
		return nil, false
	}
	v := make([]float32, wv.dim)
	s := wv.scale[i]
	for j, q := range wv.row(int(i)) {
		v[j] = s * float32(q)
	}
	return v, true
}

// Similarity returns the cosine similarity of two words' vectors, and
// false if either word is unknown.
func (wv *WordVectors) Similarity(a, b string) (float32, bool) {
	i, ok1 := wv.index[a]
	j, ok2 := wv.index[b]
	if !ok1 || !ok2 {
		return 0, false
	}
	return wv.cos(int(i), int(j)), true
}

func (wv *WordVectors) cos(i, j int) float32 {
	if wv.norm[i] == 0 || wv.norm[j] == 0 {
		return 0
	}
	var d int32
	a, b := wv.row(i), wv.row(j)
	for k := range a {
		d += int32(a[k]) * int32(b[k])
	}
	return wv.scale[i] * wv.scale[j] * float32(d) / (wv.norm[i] * wv.norm[j])
}

// Neighbor is a word returned by MostSimilar.
type Neighbor struct {
	Word       string
	Similarity float32
}

// MostSimilar returns the k words most similar to word, excluding word
// itself, best first. It scans every word, which takes a few milliseconds
// for a 100k-word table; it is meant for exploration and debugging.
func (wv *WordVectors) MostSimilar(word string, k int) ([]Neighbor, bool) {
	i, ok := wv.index[word]
	if !ok || k <= 0 {
		return nil, ok
	}
	out := make([]Neighbor, 0, k+1)
	for j := range wv.words {
		if j == int(i) {
			continue
		}
		s := wv.cos(int(i), j)
		if len(out) == k && s <= out[k-1].Similarity {
			continue
		}
		pos := sort.Search(len(out), func(x int) bool { return out[x].Similarity < s })
		out = append(out, Neighbor{})
		copy(out[pos+1:], out[pos:])
		out[pos] = Neighbor{wv.words[j], s}
		if len(out) > k {
			out = out[:k]
		}
	}
	return out, true
}

// Binary format, little endian:
//
//	magic "GOEMBWV1"
//	dim, count uint32
//	count × { len uint16, word [len]byte, prob float32, scale float32, vec [dim]int8 }
//	crc32 (IEEE) of everything above
const wvMagic = "GOEMBWV1"

// WriteTo writes wv in a compact binary format that ReadWordVectors and
// LoadWordVectors read. It implements io.WriterTo.
func (wv *WordVectors) WriteTo(w io.Writer) (int64, error) {
	cw := &countingWriter{w: w}
	h := crc32.NewIEEE()
	bw := bufio.NewWriterSize(io.MultiWriter(cw, h), 1<<16)
	var b [8]byte
	bw.WriteString(wvMagic)
	binary.LittleEndian.PutUint32(b[:4], uint32(wv.dim))
	binary.LittleEndian.PutUint32(b[4:], uint32(len(wv.words)))
	bw.Write(b[:])
	for i, word := range wv.words {
		if len(word) > math.MaxUint16 {
			return cw.n, fmt.Errorf("embedding: word %d too long", i)
		}
		binary.LittleEndian.PutUint16(b[:2], uint16(len(word)))
		bw.Write(b[:2])
		bw.WriteString(word)
		binary.LittleEndian.PutUint32(b[:4], math.Float32bits(wv.prob[i]))
		binary.LittleEndian.PutUint32(b[4:], math.Float32bits(wv.scale[i]))
		bw.Write(b[:])
		for _, q := range wv.row(i) {
			bw.WriteByte(byte(q))
		}
	}
	if err := bw.Flush(); err != nil {
		return cw.n, err
	}
	binary.LittleEndian.PutUint32(b[:4], h.Sum32())
	_, err := cw.Write(b[:4])
	return cw.n, err
}

type countingWriter struct {
	w io.Writer
	n int64
}

func (c *countingWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += int64(n)
	return n, err
}

// LoadWordVectors reads a file written by WordVectors.WriteTo.
func LoadWordVectors(path string) (*WordVectors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadWordVectors(f)
}

// ReadWordVectors reads data written by WordVectors.WriteTo. It returns an
// error wrapping ErrCorruptWordVectors if the data is invalid; it never
// panics on bad input.
func ReadWordVectors(r io.Reader) (*WordVectors, error) {
	buf, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	corrupt := func(format string, args ...any) error {
		return fmt.Errorf("%w: %s", ErrCorruptWordVectors, fmt.Sprintf(format, args...))
	}
	if len(buf) < len(wvMagic)+8+4 || string(buf[:len(wvMagic)]) != wvMagic {
		return nil, corrupt("bad header")
	}
	body, sum := buf[:len(buf)-4], binary.LittleEndian.Uint32(buf[len(buf)-4:])
	if crc32.ChecksumIEEE(body) != sum {
		return nil, corrupt("checksum mismatch")
	}
	p := body[len(wvMagic):]
	dim := int(binary.LittleEndian.Uint32(p))
	n := int(binary.LittleEndian.Uint32(p[4:]))
	p = p[8:]
	if dim < 1 || dim > 1<<16 {
		return nil, corrupt("bad dimension %d", dim)
	}
	if n < 1 || n > len(p)/(2+8+dim) {
		return nil, corrupt("bad word count %d", n)
	}
	wv := &WordVectors{
		dim:   dim,
		words: make([]string, n),
		prob:  make([]float32, n),
		scale: make([]float32, n),
		data:  make([]int8, n*dim),
	}
	seen := make(map[string]bool, n)
	for i := 0; i < n; i++ {
		if len(p) < 2 {
			return nil, corrupt("truncated at word %d", i)
		}
		l := int(binary.LittleEndian.Uint16(p))
		p = p[2:]
		if l == 0 || len(p) < l+8+dim {
			return nil, corrupt("truncated at word %d", i)
		}
		w := string(p[:l])
		if !utf8.ValidString(w) || seen[w] {
			return nil, corrupt("invalid or duplicate word %q", w)
		}
		seen[w] = true
		wv.words[i] = w
		p = p[l:]
		prob := math.Float32frombits(binary.LittleEndian.Uint32(p))
		scale := math.Float32frombits(binary.LittleEndian.Uint32(p[4:]))
		if !(prob > 0 && prob <= 1) || !(scale >= 0) || math.IsInf(float64(scale), 0) {
			return nil, corrupt("bad probability or scale for %q", w)
		}
		wv.prob[i], wv.scale[i] = prob, scale
		p = p[8:]
		for j, b := range p[:dim] {
			wv.data[i*dim+j] = int8(b)
		}
		p = p[dim:]
	}
	if len(p) != 0 {
		return nil, corrupt("%d trailing bytes", len(p))
	}
	wv.finish()
	return wv, nil
}
