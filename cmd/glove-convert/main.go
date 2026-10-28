// Command glove-convert converts GloVe word vectors from their text format
// into the compact binary format read by embedding.LoadWordVectors.
//
// It keeps the most frequent words, removes the mean and dominant
// directions ("All-but-the-Top"), quantizes to int8 and writes the result
// atomically. It reads plain .txt files or a member of the official .zip
// downloads directly, so the archive does not need to be unpacked.
//
// Usage:
//
//	glove-convert -in glove.6B.zip -member glove.6B.300d.txt -out glove-300d-100k.wv
//	glove-convert -in glove.6B.100d.txt -out glove-100d.wv -max-words 50000 -check car,king
package main

import (
	"archive/zip"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	embedding "github.com/brimstone/go-embedding"
)

func main() {
	in := flag.String("in", "", "GloVe .txt file, or .zip archive containing one (required)")
	member := flag.String("member", "", "file inside the .zip to convert (required if it holds more than one .txt)")
	out := flag.String("out", "", "output file (required)")
	maxWords := flag.Int("max-words", 100000, "keep only the N most frequent words (0 = all)")
	components := flag.Int("components", 0, "dominant components to remove (0 = dimension/100)")
	noPost := flag.Bool("no-postprocess", false, "disable mean and component removal")
	check := flag.String("check", "car,king,computer", "comma-separated words whose nearest neighbours are printed after converting")
	flag.Parse()
	if *in == "" || *out == "" {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(*in, *member, *out, *check, embedding.GloVeOptions{
		MaxWords:         *maxWords,
		Components:       *components,
		NoPostProcessing: *noPost,
	}); err != nil {
		fmt.Fprintln(os.Stderr, "glove-convert:", err)
		os.Exit(1)
	}
}

func run(in, member, out, check string, opts embedding.GloVeOptions) error {
	start := time.Now()
	r, closeIn, err := open(in, member)
	if err != nil {
		return err
	}
	wv, err := embedding.ReadGloVe(r, opts)
	closeIn()
	if err != nil {
		return err
	}
	parsed := time.Since(start)

	tmp, err := os.CreateTemp(filepath.Dir(out), ".glove-convert-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	n, err := wv.WriteTo(tmp)
	if err == nil {
		err = tmp.Chmod(0o644) // CreateTemp uses 0600
	}
	if err == nil {
		err = tmp.Sync()
	}
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.Rename(tmp.Name(), out)
	}
	if err != nil {
		return err
	}
	fmt.Printf("wrote %s: %d words × %d dims, %.1f MB (parsed in %s, total %s)\n",
		out, wv.Len(), wv.Dim(), float64(n)/1e6, parsed.Round(time.Millisecond), time.Since(start).Round(time.Millisecond))

	for _, w := range strings.Split(check, ",") {
		if w = strings.TrimSpace(w); w == "" {
			continue
		}
		nb, ok := wv.MostSimilar(w, 8)
		if !ok {
			fmt.Printf("  %-12s (not in vocabulary)\n", w)
			continue
		}
		parts := make([]string, len(nb))
		for i, x := range nb {
			parts[i] = fmt.Sprintf("%s %.2f", x.Word, x.Similarity)
		}
		fmt.Printf("  %-12s %s\n", w, strings.Join(parts, ", "))
	}
	return nil
}

// open returns a reader for a .txt file or a member of a .zip archive.
func open(path, member string) (io.Reader, func(), error) {
	if !strings.EqualFold(filepath.Ext(path), ".zip") {
		f, err := os.Open(path)
		if err != nil {
			return nil, nil, err
		}
		return f, func() { f.Close() }, nil
	}
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, nil, err
	}
	var txt []*zip.File
	for _, f := range zr.File {
		if f.Name == member {
			txt = []*zip.File{f}
			break
		}
		if member == "" && strings.HasSuffix(f.Name, ".txt") {
			txt = append(txt, f)
		}
	}
	if len(txt) != 1 {
		names := make([]string, len(zr.File))
		for i, f := range zr.File {
			names[i] = f.Name
		}
		zr.Close()
		if member != "" {
			return nil, nil, fmt.Errorf("%s has no member %q (members: %s)", path, member, strings.Join(names, ", "))
		}
		return nil, nil, errors.New(path + " contains several .txt files; choose one with -member: " + strings.Join(names, ", "))
	}
	rc, err := txt[0].Open()
	if err != nil {
		zr.Close()
		return nil, nil, err
	}
	return rc, func() { rc.Close(); zr.Close() }, nil
}
