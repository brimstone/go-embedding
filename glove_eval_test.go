package embedding_test

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	embedding "github.com/brimstone/go-embedding"
	"github.com/brimstone/go-embedding/glove"
)

// The tests in this file evaluate VectorEmbedder on real GloVe vectors: the
// bundled glove package, plus any files produced by
// cmd/glove-convert that EMBEDDING_GLOVE lists (separated by the OS path
// list separator).

type vectorSet struct {
	name    string
	wv      *embedding.WordVectors
	bundled bool
}

func vectorSets(t testing.TB) []vectorSet {
	t.Helper()
	wv, err := glove.WordVectors()
	if err != nil {
		t.Fatal(err)
	}
	sets := []vectorSet{{"glove", wv, true}}
	if list := os.Getenv("EMBEDDING_GLOVE"); list != "" {
		for _, path := range filepath.SplitList(list) {
			wv, err := embedding.LoadWordVectors(path)
			if err != nil {
				t.Fatal(err)
			}
			sets = append(sets, vectorSet{strings.TrimSuffix(filepath.Base(path), ".wv"), wv, false})
		}
	}
	return sets
}

func TestGloVeWordSimilarity(t *testing.T) {
	related := [][2]string{
		{"car", "automobile"}, {"doctor", "physician"}, {"happy", "glad"}, {"buy", "purchase"},
		{"big", "large"}, {"begin", "start"}, {"error", "mistake"}, {"fast", "quick"},
	}
	unrelated := [][2]string{
		{"car", "banana"}, {"doctor", "guitar"}, {"happy", "tractor"}, {"buy", "volcano"},
		{"big", "sermon"}, {"begin", "lettuce"}, {"error", "giraffe"}, {"fast", "pottery"},
	}
	for _, set := range vectorSets(t) {
		minRel, maxUnrel := float32(1), float32(-1)
		for _, p := range related {
			s, ok := set.wv.Similarity(p[0], p[1])
			if !ok {
				t.Fatalf("%s: %v not in vocabulary", set.name, p)
			}
			minRel = min(minRel, s)
		}
		for _, p := range unrelated {
			s, _ := set.wv.Similarity(p[0], p[1])
			maxUnrel = max(maxUnrel, s)
		}
		car, _ := set.wv.Similarity("car", "automobile")
		t.Logf("%-20s car/automobile %.3f, weakest related %.3f, strongest unrelated %.3f", set.name, car, minRel, maxUnrel)
		if minRel <= maxUnrel {
			t.Errorf("%s: related pairs not separated from unrelated: %.3f <= %.3f", set.name, minRel, maxUnrel)
		}
		if car < 0.4 {
			t.Errorf("%s: car/automobile similarity %.3f", set.name, car)
		}
	}
}

// evalDocs and evalQueries form a small retrieval benchmark. Each query
// paraphrases one document while avoiding its content words, which is
// exactly what lexical matching cannot handle.
var evalDocs = map[string]string{
	"car-maint":  "Changing the engine oil every five thousand miles keeps your car running smoothly.",
	"car-buy":    "Before buying a used vehicle, have a mechanic inspect the brakes and transmission.",
	"flight":     "Airlines often overbook flights, so arrive at the airport early to keep your seat.",
	"bread":      "Knead the dough for ten minutes, then let it rise until it doubles in size.",
	"pasta":      "Boil the noodles in salted water and drain them while they are still firm.",
	"fever":      "If your fever lasts more than three days, see a physician.",
	"teeth":      "Brush your teeth twice a day and floss to prevent cavities.",
	"saving":     "Put a portion of every paycheck into a savings account for emergencies.",
	"investing":  "Index funds spread your money across many stocks and keep fees low.",
	"garden":     "Water tomato plants deeply in the morning so the soil stays moist.",
	"dog":        "Puppies need several short walks a day and plenty of chew toys.",
	"cat":        "Kittens sleep up to twenty hours daily and love to chase string.",
	"computer":   "Restart your laptop if programs freeze or the screen stops responding.",
	"passwords":  "Use a different password for every website and enable two-factor login.",
	"weather":    "Thunderstorms bring heavy rain, lightning and strong gusts of wind.",
	"job":        "Tailor your resume to each position and prepare examples for the interview.",
	"exercise":   "Jogging three times a week strengthens your heart and lungs.",
	"insomnia":   "Avoid caffeine in the evening if you have trouble falling asleep.",
	"recycling":  "Rinse bottles and cans before putting them in the blue bin.",
	"childbirth": "Pregnant women should take folic acid and attend regular prenatal checkups.",
}

var evalQueries = map[string]string{
	"how often should I service my automobile motor lubricant": "car-maint",
	"what to check when purchasing a secondhand car":           "car-buy",
	"tips for reaching the terminal before departure":          "flight",
	"how do I make a homemade loaf":                            "bread",
	"cooking spaghetti al dente":                               "pasta",
	"when to visit a doctor about a high temperature":          "fever",
	"dental hygiene to avoid tooth decay":                      "teeth",
	"building a rainy day fund from my salary":                 "saving",
	"low cost diversified investing in shares":                 "investing",
	"irrigating vegetables":                                    "garden",
	"caring for a young dog":                                   "dog",
	"how much do young cats nap":                               "cat",
	"my computer is hanging and unresponsive":                  "computer",
	"keeping online accounts safe from hackers":                "passwords",
	"what happens during a storm":                              "weather",
	"how to get hired":                                         "job",
	"running for cardiovascular fitness":                       "exercise",
	"I can't get to sleep after drinking coffee":               "insomnia",
	"what to do with empty containers and tins":                "recycling",
	"vitamins for expecting mothers":                           "childbirth",
}

type evalResult struct{ hit1, hit3, mrr float64 }

func evaluate(t testing.TB, e embedding.Embedder) evalResult {
	t.Helper()
	ctx := context.Background()
	ids := make([]string, 0, len(evalDocs))
	for id := range evalDocs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	texts := make([]string, len(ids))
	for i, id := range ids {
		texts[i] = evalDocs[id]
	}
	docs, err := e.Embed(ctx, texts)
	if err != nil {
		t.Fatal(err)
	}
	var r evalResult
	for q, want := range evalQueries {
		qv, err := e.Embed(ctx, []string{q})
		if err != nil {
			t.Fatal(err)
		}
		type scored struct {
			id string
			s  float32
		}
		ranked := make([]scored, len(ids))
		for i, id := range ids {
			ranked[i] = scored{id, embedding.Cosine(qv[0], docs[i])}
		}
		sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].s > ranked[j].s })
		for rank, s := range ranked {
			if s.id == want {
				if rank == 0 {
					r.hit1++
				}
				if rank < 3 {
					r.hit3++
				}
				r.mrr += 1 / float64(rank+1)
				break
			}
		}
	}
	n := float64(len(evalQueries))
	return evalResult{r.hit1 / n, r.hit3 / n, r.mrr / n}
}

func TestGloVeRetrieval(t *testing.T) {
	var corpus []string
	for _, d := range evalDocs {
		corpus = append(corpus, d)
	}
	report := func(name string, e embedding.Embedder) evalResult {
		r := evaluate(t, e)
		t.Logf("%-40s hit@1 %.2f  hit@3 %.2f  MRR %.3f", name, r.hit1, r.hit3, r.mrr)
		return r
	}

	h, err := embedding.NewHashEmbedder(embedding.DefaultHashOptions())
	if err != nil {
		t.Fatal(err)
	}
	hash := report("hash (lexical baseline)", h)
	for _, set := range vectorSets(t) {
		for _, a := range []float64{1e-4, 1e-3, 1e-2} {
			e, err := embedding.NewVectorEmbedder(set.wv, embedding.SIFOptions{A: a})
			if err != nil {
				t.Fatal(err)
			}
			r := report(set.name+" SIF a="+strconv.FormatFloat(a, 'g', -1, 64), e)
			report(set.name+"   + common component", e.WithCommonComponent(corpus))
			if set.bundled && a == 1e-3 && r.mrr <= hash.mrr+0.2 {
				t.Errorf("%s: default MRR %.3f not clearly better than hash %.3f", set.name, r.mrr, hash.mrr)
			}
		}
		unweighted, _ := embedding.NewVectorEmbedder(set.wv, embedding.SIFOptions{A: 1e6})
		report(set.name+" plain average", unweighted)
	}
}
