package embedding

import (
	"math/rand"
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"
)

var vocab = strings.Fields(`the quick brown fox jumps over lazy dog vector database
embedding chunk text search query semantic similarity golang library storage index
retrieval document paragraph sentence word token cosine distance neighbour graph`)

// genText returns deterministic prose of roughly n bytes with sentences,
// lines and paragraphs.
func genText(seed int64, n int) string {
	rng := rand.New(rand.NewSource(seed))
	var b strings.Builder
	for b.Len() < n {
		words := 5 + rng.Intn(15)
		for i := range words {
			if i > 0 {
				b.WriteByte(' ')
			}
			b.WriteString(vocab[rng.Intn(len(vocab))])
		}
		switch r := rng.Intn(10); {
		case r == 0:
			b.WriteString(".\n\n")
		case r == 1:
			b.WriteString(".\n")
		case r == 2:
			b.WriteString("? ")
		default:
			b.WriteString(". ")
		}
	}
	return b.String()
}

// checkChunks verifies the Chunker contract for text.
func checkChunks(t *testing.T, text string, chunks []Chunk, maxRunes int) {
	t.Helper()
	covered := make([]bool, len(text))
	prevStart, prevEnd := -1, -1
	for i, c := range chunks {
		if c.Index != i {
			t.Fatalf("chunk %d has Index %d", i, c.Index)
		}
		if c.Start < 0 || c.End > len(text) || c.Start >= c.End {
			t.Fatalf("chunk %d has bad offsets [%d,%d) for len %d", i, c.Start, c.End, len(text))
		}
		if text[c.Start:c.End] != c.Text {
			t.Fatalf("chunk %d: text[%d:%d] = %q, Text = %q", i, c.Start, c.End, text[c.Start:c.End], c.Text)
		}
		if strings.TrimSpace(c.Text) != c.Text || c.Text == "" {
			t.Fatalf("chunk %d not trimmed or empty: %q", i, c.Text)
		}
		if n := utf8.RuneCountInString(c.Text); n > maxRunes {
			t.Fatalf("chunk %d has %d runes, max %d", i, n, maxRunes)
		}
		if c.Start <= prevStart || c.End <= prevEnd {
			t.Fatalf("chunk %d [%d,%d) not after previous [%d,%d)", i, c.Start, c.End, prevStart, prevEnd)
		}
		prevStart, prevEnd = c.Start, c.End
		for j := c.Start; j < c.End; j++ {
			covered[j] = true
		}
	}
	for i, r := range text {
		if !unicode.IsSpace(r) && !covered[i] {
			t.Fatalf("non-space rune %q at byte %d not covered by any chunk", r, i)
		}
	}
}
