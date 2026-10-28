package embedding

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
)

// Handler is an http.Handler serving an OpenAI-compatible embeddings
// endpoint backed by one or more Embedders. Mount it wherever clients
// expect it, typically:
//
//	mux.Handle("POST /v1/embeddings", &embedding.Handler{
//		Models: map[string]embedding.Embedder{"hash-512": e},
//	})
//
// It accepts POST requests with a JSON body of the form
//
//	{"model": "hash-512", "input": "text" | ["text", ...],
//	 "encoding_format": "float" | "base64", "dimensions": 512}
//
// and responds with
//
//	{"object": "list", "model": "hash-512",
//	 "data": [{"object": "embedding", "index": 0, "embedding": [...]}],
//	 "usage": {"prompt_tokens": N, "total_tokens": N}}
//
// Errors use the OpenAI shape {"error": {"message", "type", "param", "code"}}.
// Token-array inputs are not supported. Usage counts words, as an
// approximation of tokens.
type Handler struct {
	// Models maps model names to the Embedder serving them. Required.
	// If it has exactly one entry, requests may omit "model".
	Models map[string]Embedder
	// MaxInputs limits the number of texts per request. Default 2048.
	MaxInputs int
	// MaxBodyBytes limits the request body size. Default 8 MiB.
	MaxBodyBytes int64
}

type handlerRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format"`
	Dimensions     int             `json:"dimensions"`
}

type handlerEmbedding struct {
	Object    string `json:"object"`
	Index     int    `json:"index"`
	Embedding any    `json:"embedding"` // []float32 or base64 string
}

type handlerResponse struct {
	Object string             `json:"object"`
	Data   []handlerEmbedding `json:"data"`
	Model  string             `json:"model"`
	Usage  handlerUsage       `json:"usage"`
}

type handlerUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// requestError is an error reported to the client.
type requestError struct {
	status     int
	typ, param string
	code, msg  string
}

func (e *requestError) Error() string { return e.msg }

func invalid(param, format string, args ...any) *requestError {
	return &requestError{status: http.StatusBadRequest, typ: "invalid_request_error", param: param, msg: fmt.Sprintf(format, args...)}
}

// ServeHTTP implements http.Handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, &requestError{status: http.StatusMethodNotAllowed, typ: "invalid_request_error", msg: "method not allowed; use POST"})
		return
	}
	resp, err := h.serve(r)
	if err != nil {
		var re *requestError
		if !errors.As(err, &re) {
			if r.Context().Err() != nil {
				return // client went away
			}
			re = &requestError{status: http.StatusInternalServerError, typ: "server_error", msg: err.Error()}
		}
		writeError(w, re)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (h *Handler) serve(r *http.Request) (*handlerResponse, error) {
	maxBody := h.MaxBodyBytes
	if maxBody <= 0 {
		maxBody = 8 << 20
	}
	maxInputs := h.MaxInputs
	if maxInputs <= 0 {
		maxInputs = 2048
	}

	var req handlerRequest
	dec := json.NewDecoder(http.MaxBytesReader(nil, r.Body, maxBody))
	if err := dec.Decode(&req); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			return nil, &requestError{status: http.StatusRequestEntityTooLarge, typ: "invalid_request_error", msg: fmt.Sprintf("request body exceeds %d bytes", maxBody)}
		}
		return nil, invalid("", "invalid JSON body: %v", err)
	}

	name, e, err := h.model(req.Model)
	if err != nil {
		return nil, err
	}
	texts, err := parseInput(req.Input)
	if err != nil {
		return nil, err
	}
	if len(texts) > maxInputs {
		return nil, invalid("input", "too many inputs: %d > %d", len(texts), maxInputs)
	}
	switch req.EncodingFormat {
	case "", "float", "base64":
	default:
		return nil, invalid("encoding_format", "unsupported encoding_format %q; use \"float\" or \"base64\"", req.EncodingFormat)
	}
	if d := e.Dimension(); req.Dimensions != 0 && d != 0 && req.Dimensions != d {
		return nil, invalid("dimensions", "model %q produces %d dimensions, not %d", name, d, req.Dimensions)
	}

	vecs, err := e.Embed(r.Context(), texts)
	if err != nil {
		return nil, err
	}
	if len(vecs) != len(texts) {
		return nil, fmt.Errorf("embedder returned %d vectors for %d inputs", len(vecs), len(texts))
	}

	resp := &handlerResponse{Object: "list", Model: name, Data: make([]handlerEmbedding, len(vecs))}
	for i, v := range vecs {
		var emb any = v
		if req.EncodingFormat == "base64" {
			emb = encodeBase64(v)
		}
		resp.Data[i] = handlerEmbedding{Object: "embedding", Index: i, Embedding: emb}
		resp.Usage.PromptTokens += countWords(texts[i])
	}
	resp.Usage.TotalTokens = resp.Usage.PromptTokens
	return resp, nil
}

// model resolves the requested model name.
func (h *Handler) model(name string) (string, Embedder, error) {
	if len(h.Models) == 0 {
		return "", nil, errors.New("handler has no models configured")
	}
	if name == "" {
		if len(h.Models) == 1 {
			for n, e := range h.Models {
				return n, e, nil
			}
		}
		return "", nil, invalid("model", "model is required")
	}
	e, ok := h.Models[name]
	if !ok {
		names := make([]string, 0, len(h.Models))
		for n := range h.Models {
			names = append(names, n)
		}
		sort.Strings(names)
		return "", nil, &requestError{status: http.StatusNotFound, typ: "invalid_request_error", param: "model", code: "model_not_found",
			msg: fmt.Sprintf("model %q does not exist; available: %s", name, strings.Join(names, ", "))}
	}
	return name, e, nil
}

// parseInput accepts a string or an array of strings.
func parseInput(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, invalid("input", "input is required")
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		return []string{one}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return nil, invalid("input", "input must be a string or an array of strings")
	}
	if len(many) == 0 {
		return nil, invalid("input", "input must not be empty")
	}
	return many, nil
}

// encodeBase64 encodes v as little-endian float32s, as OpenAI does.
func encodeBase64(v []float32) string {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return base64.StdEncoding.EncodeToString(b)
}

func countWords(s string) int {
	n, in := 0, false
	for _, r := range s {
		w := isWordRune(r)
		if w && !in {
			n++
		}
		in = w
	}
	return n
}

func writeError(w http.ResponseWriter, e *requestError) {
	type body struct {
		Message string  `json:"message"`
		Type    string  `json:"type"`
		Param   *string `json:"param"`
		Code    *string `json:"code"`
	}
	b := body{Message: e.msg, Type: e.typ}
	if e.param != "" {
		b.Param = &e.param
	}
	if e.code != "" {
		b.Code = &e.code
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.status)
	_ = json.NewEncoder(w).Encode(map[string]body{"error": b})
}
