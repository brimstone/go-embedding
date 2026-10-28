package embedding

import (
	"errors"
	"strings"
	"testing"
	"unicode/utf8"
)

func chunkers(t testing.TB, size, overlap int) map[string]Chunker {
	t.Helper()
	f, err := NewFixedChunker(size, overlap)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewRecursiveChunker(size, overlap)
	if err != nil {
		t.Fatal(err)
	}
	return map[string]Chunker{"fixed": f, "recursive": r}
}

func TestChunkerValidation(t *testing.T) {
	for _, tc := range []struct{ size, overlap int }{{0, 0}, {-1, 0}, {10, -1}, {10, 10}, {10, 11}} {
		if _, err := NewFixedChunker(tc.size, tc.overlap); !errors.Is(err, ErrInvalidChunkSize) {
			t.Errorf("NewFixedChunker(%d, %d) err = %v", tc.size, tc.overlap, err)
		}
		if _, err := NewRecursiveChunker(tc.size, tc.overlap); !errors.Is(err, ErrInvalidChunkSize) {
			t.Errorf("NewRecursiveChunker(%d, %d) err = %v", tc.size, tc.overlap, err)
		}
	}
	if _, err := NewRecursiveChunker(10, 0, "\n", ""); err == nil {
		t.Error("empty separator accepted")
	}
}

func TestChunkEmptyAndWhitespace(t *testing.T) {
	for name, c := range chunkers(t, 10, 2) {
		for _, s := range []string{"", " ", "\n\n\t  \n"} {
			if got := c.Chunk(s); len(got) != 0 {
				t.Errorf("%s: Chunk(%q) = %v, want none", name, s, got)
			}
		}
	}
}

func TestChunkShortText(t *testing.T) {
	for name, c := range chunkers(t, 100, 10) {
		got := c.Chunk("  hello world  ")
		if len(got) != 1 || got[0].Text != "hello world" || got[0].Start != 2 || got[0].End != 13 {
			t.Errorf("%s: got %+v", name, got)
		}
	}
}

func TestChunkInvariants(t *testing.T) {
	texts := map[string]string{
		"prose":   genText(1, 20000),
		"unicode": strings.Repeat("Größe naïve café 日本語のテキスト 🙂 emoji. ", 200),
		"nospace": strings.Repeat("x", 1000),
		"mixed":   "a\n\nb\nc. d? e! f; g: h, i j " + strings.Repeat("long", 100) + "\n\n" + genText(2, 3000),
	}
	for _, sz := range []struct{ size, overlap int }{{50, 0}, {50, 10}, {200, 50}, {1, 0}, {7, 3}} {
		for name, c := range chunkers(t, sz.size, sz.overlap) {
			for tname, text := range texts {
				chunks := c.Chunk(text)
				if len(chunks) == 0 {
					t.Fatalf("%s/%s: no chunks", name, tname)
				}
				checkChunks(t, text, chunks, sz.size)
			}
		}
	}
}

func TestFixedChunkerBreaksAtWords(t *testing.T) {
	c, _ := NewFixedChunker(12, 0)
	got := c.Chunk("alpha beta gamma delta epsilon")
	want := []string{"alpha beta", "gamma delta", "epsilon"}
	if len(got) != len(want) {
		t.Fatalf("got %d chunks %+v, want %v", len(got), got, want)
	}
	for i := range want {
		if got[i].Text != want[i] {
			t.Errorf("chunk %d = %q, want %q", i, got[i].Text, want[i])
		}
	}
}

func TestFixedChunkerOverlap(t *testing.T) {
	c, _ := NewFixedChunker(20, 8)
	text := "one two three four five six seven eight nine ten eleven twelve"
	chunks := c.Chunk(text)
	checkChunks(t, text, chunks, 20)
	for i := 1; i < len(chunks); i++ {
		if chunks[i].Start >= chunks[i-1].End {
			t.Errorf("chunks %d and %d do not overlap: %+v %+v", i-1, i, chunks[i-1], chunks[i])
		}
	}
}

func TestRecursiveChunkerPrefersParagraphs(t *testing.T) {
	c, _ := NewRecursiveChunker(30, 0)
	text := "First paragraph is here.\n\nSecond paragraph is here.\n\nThird one."
	got := c.Chunk(text)
	want := []string{"First paragraph is here.", "Second paragraph is here.", "Third one."}
	if len(got) != len(want) {
		t.Fatalf("got %+v", got)
	}
	for i := range want {
		if got[i].Text != want[i] {
			t.Errorf("chunk %d = %q, want %q", i, got[i].Text, want[i])
		}
	}
}

func TestRecursiveChunkerSentences(t *testing.T) {
	c, _ := NewRecursiveChunker(30, 0)
	text := "Cats sleep a lot. Dogs bark loudly. Birds sing at dawn. Fish swim."
	got := c.Chunk(text)
	checkChunks(t, text, got, 30)
	for _, ch := range got {
		if !strings.HasSuffix(ch.Text, ".") {
			t.Errorf("chunk %q does not end at a sentence boundary", ch.Text)
		}
	}
}

func TestRecursiveChunkerOverlap(t *testing.T) {
	c, _ := NewRecursiveChunker(30, 15)
	text := "aa bb cc dd ee ff gg hh ii jj kk ll mm nn oo pp qq rr ss tt"
	chunks := c.Chunk(text)
	checkChunks(t, text, chunks, 30)
	if len(chunks) < 2 {
		t.Fatalf("want multiple chunks, got %+v", chunks)
	}
	for i := 1; i < len(chunks); i++ {
		ov := chunks[i-1].End - chunks[i].Start
		if ov <= 0 || ov > 15 {
			t.Errorf("overlap between %d and %d is %d bytes, want (0,15]", i-1, i, ov)
		}
	}
}

func TestRecursiveChunkerCustomSeparators(t *testing.T) {
	c, _ := NewRecursiveChunker(5, 0, "|")
	got := c.Chunk("ab|cd|efghijkl")
	want := []string{"ab|", "cd|", "efghi", "jkl"}
	if len(got) != len(want) {
		t.Fatalf("got %+v", got)
	}
	for i := range want {
		if got[i].Text != want[i] {
			t.Errorf("chunk %d = %q, want %q", i, got[i].Text, want[i])
		}
	}
}

func FuzzChunkers(f *testing.F) {
	f.Add("hello world. this is a test.\n\nnew para", 10, 3)
	f.Add("日本語 テキスト。 🙂🙂🙂", 3, 1)
	f.Add("\xff\xfe invalid utf8 \x80", 4, 0)
	f.Fuzz(func(t *testing.T, text string, size, overlap int) {
		size = 1 + abs(size)%200
		overlap = abs(overlap) % size
		for _, c := range chunkers(t, size, overlap) {
			chunks := c.Chunk(text)
			if utf8.ValidString(text) {
				checkChunks(t, text, chunks, size)
			}
		}
	})
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
