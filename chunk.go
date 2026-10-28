package embedding

import (
	"errors"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Chunk is a contiguous piece of a source text.
type Chunk struct {
	// Index is the chunk's position in the sequence returned by a Chunker.
	Index int
	// Text is the chunk's content, with surrounding whitespace removed.
	// It always equals source[Start:End].
	Text string
	// Start and End are the byte offsets of Text within the source.
	Start, End int
}

// Chunker splits a text into chunks. Implementations must be safe for
// concurrent use. Chunks are returned in source order, never contain only
// whitespace, and every non-whitespace character of the source appears in
// at least one chunk.
type Chunker interface {
	Chunk(text string) []Chunk
}

// ErrInvalidChunkSize is returned when a chunker is configured with an
// invalid size or overlap.
var ErrInvalidChunkSize = errors.New("embedding: invalid chunk size")

func validateSize(size, overlap int) error {
	if size <= 0 {
		return fmt.Errorf("%w: size %d must be positive", ErrInvalidChunkSize, size)
	}
	if overlap < 0 || overlap >= size {
		return fmt.Errorf("%w: overlap %d must be in [0, %d)", ErrInvalidChunkSize, overlap, size)
	}
	return nil
}

// appendChunk trims whitespace from text[start:end] and appends the result
// to out. Empty results and results contained in the previous chunk are
// dropped; a result containing the previous chunk replaces it.
func appendChunk(out []Chunk, text string, start, end int) []Chunk {
	for start < end {
		r, n := utf8.DecodeRuneInString(text[start:end])
		if !unicode.IsSpace(r) {
			break
		}
		start += n
	}
	for end > start {
		r, n := utf8.DecodeLastRuneInString(text[start:end])
		if !unicode.IsSpace(r) {
			break
		}
		end -= n
	}
	if start == end {
		return out
	}
	if n := len(out); n > 0 {
		prev := out[n-1]
		if end <= prev.End {
			return out // contained in the previous chunk
		}
		if start <= prev.Start {
			// Trimming made this chunk contain the previous one.
			out[n-1] = Chunk{Index: n - 1, Text: text[start:end], Start: start, End: end}
			return out
		}
	}
	return append(out, Chunk{Index: len(out), Text: text[start:end], Start: start, End: end})
}

// advance returns the byte offset n runes after i, capped at len(s).
func advance(s string, i, n int) int {
	for ; n > 0 && i < len(s); n-- {
		_, w := utf8.DecodeRuneInString(s[i:])
		i += w
	}
	return i
}

// retreat returns the byte offset n runes before i, capped at 0.
func retreat(s string, i, n int) int {
	for ; n > 0 && i > 0; n-- {
		_, w := utf8.DecodeLastRuneInString(s[:i])
		i -= w
	}
	return i
}

// FixedChunker splits text into windows of at most Size runes, with
// consecutive windows sharing up to Overlap runes. Where possible windows
// end at whitespace and start at the beginning of a word, so words are not
// cut in half.
type FixedChunker struct {
	size, overlap int
}

// NewFixedChunker returns a FixedChunker producing chunks of at most size
// runes with up to overlap runes shared between neighbours. It requires
// size > 0 and 0 <= overlap < size.
func NewFixedChunker(size, overlap int) (*FixedChunker, error) {
	if err := validateSize(size, overlap); err != nil {
		return nil, err
	}
	return &FixedChunker{size: size, overlap: overlap}, nil
}

// Chunk implements Chunker.
func (c *FixedChunker) Chunk(text string) []Chunk {
	var out []Chunk
	start := 0
	for start < len(text) {
		end := advance(text, start, c.size)
		if end < len(text) {
			// Break at the last whitespace if it is in the second half of
			// the window; otherwise cut mid-word rather than make a tiny
			// chunk.
			if i := strings.LastIndexFunc(text[start:end], unicode.IsSpace); i > (end-start)/2 {
				end = start + i
			}
		}
		out = appendChunk(out, text, start, end)
		if end >= len(text) {
			break
		}
		next := retreat(text, end, c.overlap)
		if next > 0 && next < end {
			// Mid-word: move forward to the next word boundary.
			prev, _ := utf8.DecodeLastRuneInString(text[:next])
			cur, _ := utf8.DecodeRuneInString(text[next:])
			if !unicode.IsSpace(prev) && !unicode.IsSpace(cur) {
				if i := strings.IndexFunc(text[next:end], unicode.IsSpace); i >= 0 {
					next += i
				}
			}
		}
		if next <= start {
			next = end
		}
		start = next
	}
	return out
}

// DefaultSeparators are the boundaries RecursiveChunker splits on, from
// strongest to weakest: paragraphs, lines, sentences, clauses and words.
var DefaultSeparators = []string{"\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " "}

// RecursiveChunker splits text on the strongest boundary that yields
// pieces of at most Size runes, recursing to weaker boundaries (and
// finally a hard cut) for pieces that are still too long. Adjacent pieces
// are then greedily merged into chunks of at most Size runes. Consecutive
// chunks share whole trailing pieces totalling at most Overlap runes.
type RecursiveChunker struct {
	size, overlap int
	seps          []string
}

// NewRecursiveChunker returns a RecursiveChunker producing chunks of at
// most size runes with up to overlap runes shared between neighbours. If
// no separators are given, DefaultSeparators is used. Separators are
// tried in order and kept at the end of the piece they terminate.
func NewRecursiveChunker(size, overlap int, separators ...string) (*RecursiveChunker, error) {
	if err := validateSize(size, overlap); err != nil {
		return nil, err
	}
	if len(separators) == 0 {
		separators = DefaultSeparators
	}
	for _, s := range separators {
		if s == "" {
			return nil, errors.New("embedding: empty separator")
		}
	}
	return &RecursiveChunker{size: size, overlap: overlap, seps: append([]string(nil), separators...)}, nil
}

// span is a piece of the source: byte offsets and rune count.
type span struct{ start, end, n int }

// Chunk implements Chunker.
func (c *RecursiveChunker) Chunk(text string) []Chunk {
	if text == "" {
		return nil
	}
	return c.merge(text, c.split(nil, text, 0, len(text), utf8.RuneCountInString(text), 0))
}

// split appends pieces of text[start:end] (n runes long) of at most c.size
// runes to dst. The pieces are contiguous and cover the whole range.
func (c *RecursiveChunker) split(dst []span, text string, start, end, n, level int) []span {
	if n <= c.size {
		return append(dst, span{start, end, n})
	}
	if level == len(c.seps) {
		for start < end {
			e := min(advance(text, start, c.size), end)
			dst = append(dst, span{start, e, utf8.RuneCountInString(text[start:e])})
			start = e
		}
		return dst
	}
	sep := c.seps[level]
	if !strings.Contains(text[start:end], sep) {
		return c.split(dst, text, start, end, n, level+1)
	}
	for start < end {
		e := end
		if i := strings.Index(text[start:end], sep); i >= 0 {
			e = start + i + len(sep)
		}
		dst = c.split(dst, text, start, e, utf8.RuneCountInString(text[start:e]), level+1)
		start = e
	}
	return dst
}

// merge greedily packs contiguous pieces into chunks of at most c.size runes.
func (c *RecursiveChunker) merge(text string, p []span) []Chunk {
	var out []Chunk
	i, j, total := 0, 0, 0
	for j < len(p) {
		for j < len(p) && (j == i || total+p[j].n <= c.size) {
			total += p[j].n
			j++
		}
		out = appendChunk(out, text, p[i].start, p[j-1].end)
		if j == len(p) {
			break
		}
		// Carry trailing pieces into the next chunk as overlap, always
		// dropping at least one piece and leaving room for p[j].
		k, ov := j, 0
		for k-1 > i && ov+p[k-1].n <= c.overlap && ov+p[k-1].n+p[j].n <= c.size {
			k--
			ov += p[k].n
		}
		i, total = k, ov
	}
	return out
}
