package embedding

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type apiResponse struct {
	Object string `json:"object"`
	Model  string `json:"model"`
	Data   []struct {
		Object    string          `json:"object"`
		Index     int             `json:"index"`
		Embedding json.RawMessage `json:"embedding"`
	} `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string  `json:"message"`
		Type    string  `json:"type"`
		Param   *string `json:"param"`
		Code    *string `json:"code"`
	} `json:"error"`
}

func post(t *testing.T, h http.Handler, body string) (int, apiResponse) {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body)))
	var r apiResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &r); err != nil {
		t.Fatalf("invalid JSON response %q: %v", rec.Body.String(), err)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q", ct)
	}
	return rec.Code, r
}

func TestHandlerFloat(t *testing.T) {
	e := newHash(t)
	h := &Handler{Models: map[string]Embedder{"hash": e}}
	code, r := post(t, h, `{"model":"hash","input":["hello world","second text here"]}`)
	if code != http.StatusOK {
		t.Fatalf("status %d: %+v", code, r.Error)
	}
	if r.Object != "list" || r.Model != "hash" || len(r.Data) != 2 {
		t.Fatalf("unexpected response %+v", r)
	}
	if r.Usage.PromptTokens != 5 || r.Usage.TotalTokens != 5 {
		t.Errorf("usage = %+v, want 5 tokens", r.Usage)
	}
	for i, d := range r.Data {
		var v []float32
		if err := json.Unmarshal(d.Embedding, &v); err != nil {
			t.Fatal(err)
		}
		want := embedOne(t, e, []string{"hello world", "second text here"}[i])
		if d.Index != i || d.Object != "embedding" || len(v) != len(want) {
			t.Fatalf("data[%d] = index %d object %q len %d", i, d.Index, d.Object, len(v))
		}
		for j := range v {
			if v[j] != want[j] {
				t.Fatalf("data[%d][%d] = %v, want %v", i, j, v[j], want[j])
			}
		}
	}
}

func TestHandlerBase64AndSingleInput(t *testing.T) {
	e := newHash(t)
	h := &Handler{Models: map[string]Embedder{"hash": e}}
	code, r := post(t, h, `{"input":"hello world","encoding_format":"base64","dimensions":512}`)
	if code != http.StatusOK {
		t.Fatalf("status %d: %+v", code, r.Error)
	}
	var s string
	if err := json.Unmarshal(r.Data[0].Embedding, &s); err != nil {
		t.Fatal(err)
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		t.Fatal(err)
	}
	want := embedOne(t, e, "hello world")
	if len(b) != 4*len(want) {
		t.Fatalf("decoded %d bytes, want %d", len(b), 4*len(want))
	}
	for i := range want {
		if got := math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:])); got != want[i] {
			t.Fatalf("[%d] = %v, want %v", i, got, want[i])
		}
	}
}

type failingEmbedder struct{ err error }

func (f failingEmbedder) Embed(context.Context, []string) ([][]float32, error) { return nil, f.err }
func (f failingEmbedder) Dimension() int                                       { return 4 }

func TestHandlerErrors(t *testing.T) {
	two := map[string]Embedder{"a": newHash(t), "b": newHash(t)}
	tests := []struct {
		name      string
		h         *Handler
		body      string
		status    int
		param     string
		code      string
		typ       string
		msgSubstr string
	}{
		{"bad json", &Handler{Models: two}, `{`, 400, "", "", "invalid_request_error", "invalid JSON"},
		{"missing model", &Handler{Models: two}, `{"input":"x"}`, 400, "model", "", "invalid_request_error", "required"},
		{"unknown model", &Handler{Models: two}, `{"model":"zz","input":"x"}`, 404, "model", "model_not_found", "invalid_request_error", "available: a, b"},
		{"missing input", &Handler{Models: two}, `{"model":"a"}`, 400, "input", "", "invalid_request_error", "required"},
		{"empty input", &Handler{Models: two}, `{"model":"a","input":[]}`, 400, "input", "", "invalid_request_error", "empty"},
		{"token input", &Handler{Models: two}, `{"model":"a","input":[1,2,3]}`, 400, "input", "", "invalid_request_error", "string"},
		{"too many", &Handler{Models: two, MaxInputs: 1}, `{"model":"a","input":["x","y"]}`, 400, "input", "", "invalid_request_error", "too many"},
		{"format", &Handler{Models: two}, `{"model":"a","input":"x","encoding_format":"int8"}`, 400, "encoding_format", "", "invalid_request_error", "unsupported"},
		{"dimensions", &Handler{Models: two}, `{"model":"a","input":"x","dimensions":3}`, 400, "dimensions", "", "invalid_request_error", "512"},
		{"too large", &Handler{Models: two, MaxBodyBytes: 10}, `{"model":"a","input":"xxxxxxxxxxxxxxxx"}`, 413, "", "", "invalid_request_error", "exceeds"},
		{"no models", &Handler{}, `{"input":"x"}`, 500, "", "", "server_error", "no models"},
		{"embed fails", &Handler{Models: map[string]Embedder{"f": failingEmbedder{errors.New("boom")}}}, `{"input":"x"}`, 500, "", "", "server_error", "boom"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			code, r := post(t, tc.h, tc.body)
			if code != tc.status {
				t.Fatalf("status %d, want %d", code, tc.status)
			}
			if r.Error == nil {
				t.Fatal("no error object")
			}
			if r.Error.Type != tc.typ || !strings.Contains(r.Error.Message, tc.msgSubstr) {
				t.Errorf("error = %+v", *r.Error)
			}
			if got := deref(r.Error.Param); got != tc.param {
				t.Errorf("param = %q, want %q", got, tc.param)
			}
			if got := deref(r.Error.Code); got != tc.code {
				t.Errorf("code = %q, want %q", got, tc.code)
			}
		})
	}
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func TestHandlerMethodNotAllowed(t *testing.T) {
	rec := httptest.NewRecorder()
	(&Handler{Models: map[string]Embedder{"a": newHash(t)}}).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != http.MethodPost {
		t.Errorf("status %d, Allow %q", rec.Code, rec.Header().Get("Allow"))
	}
}

func TestHandlerOverHTTP(t *testing.T) {
	mux := http.NewServeMux()
	mux.Handle("/v1/embeddings", &Handler{Models: map[string]Embedder{"hash": newHash(t)}})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	resp, err := http.Post(srv.URL+"/v1/embeddings", "application/json", strings.NewReader(`{"model":"hash","input":"over the wire"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var r apiResponse
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK || len(r.Data) != 1 {
		t.Fatalf("status %d, response %+v", resp.StatusCode, r)
	}
}
