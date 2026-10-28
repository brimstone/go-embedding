package embedding

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"testing"
)

// gloveText renders words and vectors in GloVe's text format.
func gloveText(words []string, vecs [][]float32) string {
	var b strings.Builder
	for i, w := range words {
		b.WriteString(w)
		for _, x := range vecs[i] {
			fmt.Fprintf(&b, " %g", x)
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// randomGloVe returns n random words of dimension d in GloVe format.
func randomGloVe(seed int64, n, d int) (string, []string, [][]float32) {
	rng := rand.New(rand.NewSource(seed))
	words := make([]string, n)
	vecs := make([][]float32, n)
	for i := range words {
		words[i] = fmt.Sprintf("w%d", i)
		vecs[i] = make([]float32, d)
		for j := range vecs[i] {
			vecs[i][j] = float32(rng.NormFloat64())
		}
	}
	return gloveText(words, vecs), words, vecs
}

func mustGloVe(t testing.TB, text string, opts GloVeOptions) *WordVectors {
	t.Helper()
	wv, err := ReadGloVe(strings.NewReader(text), opts)
	if err != nil {
		t.Fatal(err)
	}
	return wv
}

func cos64(a, b []float32) float64 { return float64(Cosine(a, b)) }

func TestReadGloVeBasic(t *testing.T) {
	text := "the 1 0 0\ncar 0 1 0.5\n\nmulti word 0 0 1\ncar 9 9 9\nlast -1 -1 -1"
	wv := mustGloVe(t, text, GloVeOptions{NoPostProcessing: true})
	if wv.Dim() != 3 || wv.Len() != 4 {
		t.Fatalf("dim %d len %d", wv.Dim(), wv.Len())
	}
	for i, w := range []string{"the", "car", "multi word", "last"} {
		if wv.Word(i) != w {
			t.Errorf("Word(%d) = %q, want %q", i, wv.Word(i), w)
		}
	}
	v, ok := wv.Vector("car") // first occurrence wins
	if !ok || math.Abs(float64(v[1])-1) > 0.01 || math.Abs(float64(v[2])-0.5) > 0.01 || v[0] != 0 {
		t.Errorf("car = %v", v)
	}
	if _, ok := wv.Vector("missing"); ok {
		t.Error("missing word found")
	}
	// Probabilities follow rank and are valid.
	var sum float64
	for i := 0; i < wv.Len(); i++ {
		p := float64(wv.prob[i])
		sum += p
		if p <= 0 || (i > 0 && wv.prob[i] >= wv.prob[i-1]) {
			t.Errorf("prob[%d] = %v not positive and decreasing", i, p)
		}
	}
	if sum > 1.0001 {
		t.Errorf("probabilities sum to %v", sum)
	}
}

func TestReadGloVeMaxWords(t *testing.T) {
	text, _, _ := randomGloVe(1, 50, 4)
	all := mustGloVe(t, text, GloVeOptions{NoPostProcessing: true})
	some := mustGloVe(t, text, GloVeOptions{MaxWords: 10, NoPostProcessing: true})
	if some.Len() != 10 || some.Word(9) != "w9" {
		t.Fatalf("len %d", some.Len())
	}
	// Probabilities are normalised over the whole file, not just kept words.
	if some.prob[0] != all.prob[0] {
		t.Errorf("prob[0] = %v with MaxWords, %v without", some.prob[0], all.prob[0])
	}
}

func TestReadGloVeErrors(t *testing.T) {
	for name, text := range map[string]string{
		"empty":     "",
		"blank":     "\n\n  \n",
		"no values": "word\n",
		"short":     "a 1 2 3\nb 1 2\n",
		"bad float": "a 1 x 3\n",
		"nan":       "a 1 NaN 3\n",
		"inf":       "a 1 +Inf 3\n",
	} {
		if _, err := ReadGloVe(strings.NewReader(text), GloVeOptions{}); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	if _, err := ReadGloVe(strings.NewReader("a 1\n"), GloVeOptions{MaxWords: -1}); err == nil {
		t.Error("negative MaxWords accepted")
	}
}

func TestReadGloVeLongLines(t *testing.T) {
	text, _, vecs := randomGloVe(2, 3, 70000) // lines longer than the read buffer
	if _, err := ReadGloVe(strings.NewReader(text), GloVeOptions{}); err == nil {
		t.Fatal("dimension above 65536 accepted")
	}
	text, _, vecs = randomGloVe(2, 3, 40000)
	wv := mustGloVe(t, text, GloVeOptions{NoPostProcessing: true})
	v, _ := wv.Vector("w2")
	if cos64(v, vecs[2]) < 0.999 {
		t.Error("long line parsed incorrectly")
	}
}

func TestQuantizationPreservesSimilarity(t *testing.T) {
	text, words, vecs := randomGloVe(3, 200, 300)
	wv := mustGloVe(t, text, GloVeOptions{NoPostProcessing: true})
	var worst float64
	for i := 0; i < 100; i++ {
		a, b := i, (i*7+3)%200
		got, _ := wv.Similarity(words[a], words[b])
		worst = math.Max(worst, math.Abs(float64(got)-cos64(vecs[a], vecs[b])))
		v, _ := wv.Vector(words[a])
		if c := cos64(v, vecs[a]); c < 0.999 {
			t.Fatalf("quantized vector cosine to original %v", c)
		}
	}
	if worst > 0.01 {
		t.Errorf("quantization changed a similarity by %v", worst)
	}
}

func TestAllButTheTop(t *testing.T) {
	// Vectors with a large shared offset and one dominant direction.
	rng := rand.New(rand.NewSource(4))
	const n, d = 500, 20
	x := make([]float32, n*d)
	for i := 0; i < n; i++ {
		big := float32(rng.NormFloat64() * 10)
		for j := 0; j < d; j++ {
			x[i*d+j] = 5 + float32(rng.NormFloat64())
		}
		x[i*d+3] += big
	}
	allButTheTop(x, d, 1)
	var mean [d]float64
	var along float64
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			mean[j] += float64(x[i*d+j]) / n
		}
		along += float64(x[i*d+3]) * float64(x[i*d+3]) / n
	}
	for j, m := range mean {
		if math.Abs(m) > 1e-3 {
			t.Errorf("mean[%d] = %v after removal", j, m)
		}
	}
	if along > 1.5 { // ~1 from the noise that remains
		t.Errorf("variance along dominant direction %v after removal", along)
	}
}

func TestTopComponents(t *testing.T) {
	c := []float64{ // eigenvalues 9, 4, 1 on axes 2, 0, 1
		4, 0, 0,
		0, 1, 0,
		0, 0, 9,
	}
	pcs := topComponents(c, 3, 2)
	if len(pcs) != 2 || math.Abs(math.Abs(pcs[0][2])-1) > 1e-6 || math.Abs(math.Abs(pcs[1][0])-1) > 1e-6 {
		t.Errorf("components = %v", pcs)
	}
}

func TestMostSimilar(t *testing.T) {
	words := []string{"car", "automobile", "banana", "vehicle"}
	vecs := [][]float32{{1, 0, 0}, {0.9, 0.1, 0}, {0, 0, 1}, {0.7, 0.5, 0}}
	wv := mustGloVe(t, gloveText(words, vecs), GloVeOptions{NoPostProcessing: true})
	nb, ok := wv.MostSimilar("car", 2)
	if !ok || len(nb) != 2 || nb[0].Word != "automobile" || nb[1].Word != "vehicle" {
		t.Errorf("MostSimilar = %+v", nb)
	}
	if nb, _ := wv.MostSimilar("car", 10); len(nb) != 3 {
		t.Errorf("k > len returned %d", len(nb))
	}
	if _, ok := wv.MostSimilar("nope", 3); ok {
		t.Error("unknown word reported ok")
	}
	if s, ok := wv.Similarity("car", "banana"); !ok || math.Abs(float64(s)) > 1e-6 {
		t.Errorf("Similarity(car, banana) = %v", s)
	}
	if _, ok := wv.Similarity("car", "nope"); ok {
		t.Error("unknown word reported ok")
	}
}

func TestWordVectorsRoundTrip(t *testing.T) {
	text, words, _ := randomGloVe(5, 300, 16)
	wv := mustGloVe(t, text, GloVeOptions{})
	var buf bytes.Buffer
	n, err := wv.WriteTo(&buf)
	if err != nil || n != int64(buf.Len()) {
		t.Fatalf("WriteTo = %d, %v (buffer %d)", n, err, buf.Len())
	}
	data := buf.Bytes()
	wv2, err := ReadWordVectors(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if wv2.Len() != wv.Len() || wv2.Dim() != wv.Dim() {
		t.Fatalf("shape changed")
	}
	for i, w := range words {
		a, _ := wv.Vector(w)
		b, _ := wv2.Vector(w)
		for j := range a {
			if a[j] != b[j] {
				t.Fatalf("vector %s changed", w)
			}
		}
		if wv.prob[i] != wv2.prob[i] {
			t.Fatalf("prob %s changed", w)
		}
	}

	corrupt := map[string][]byte{
		"empty":     nil,
		"magic":     append([]byte("XXXXXXXX"), data[8:]...),
		"truncated": data[:len(data)/2],
		"short":     data[:len(data)-1],
		"trailing":  append(append([]byte(nil), data...), 0),
	}
	for _, pos := range []int{9, 20, len(data) / 2, len(data) - 1} {
		b := append([]byte(nil), data...)
		b[pos] ^= 1
		corrupt[fmt.Sprint("flip@", pos)] = b
	}
	for name, b := range corrupt {
		if _, err := ReadWordVectors(bytes.NewReader(b)); !errors.Is(err, ErrCorruptWordVectors) {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}

func FuzzReadWordVectors(f *testing.F) {
	text, _, _ := randomGloVe(6, 5, 3)
	wv, _ := ReadGloVe(strings.NewReader(text), GloVeOptions{})
	var buf bytes.Buffer
	wv.WriteTo(&buf)
	f.Add(buf.Bytes())
	f.Add([]byte(wvMagic))
	f.Fuzz(func(t *testing.T, data []byte) {
		// Fix the checksum so the fuzzer reaches the parser.
		if len(data) >= 4 {
			body := data[:len(data)-4]
			sum := crcIEEE(body)
			data = append(append([]byte(nil), body...), byte(sum), byte(sum>>8), byte(sum>>16), byte(sum>>24))
		}
		wv, err := ReadWordVectors(bytes.NewReader(data))
		if err == nil {
			for i := 0; i < wv.Len(); i++ {
				wv.Vector(wv.Word(i))
				wv.MostSimilar(wv.Word(i), 2)
			}
		}
	})
}

// Converted files are committed to the repository, so conversion must be
// reproducible regardless of the number of CPUs.
func TestReadGloVeDeterministic(t *testing.T) {
	text, _, _ := randomGloVe(7, 3000, 50)
	convert := func(procs int) []byte {
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
		var buf bytes.Buffer
		if _, err := mustGloVe(t, text, GloVeOptions{Components: 3}).WriteTo(&buf); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}
	want := convert(1)
	for _, procs := range []int{2, 3, 8, 32} {
		if !bytes.Equal(convert(procs), want) {
			t.Fatalf("output with GOMAXPROCS=%d differs from GOMAXPROCS=1", procs)
		}
	}
}
