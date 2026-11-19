/* ═══════════════════════════════════════════════════════════════════
   TriadLang Native LLM — Tokenizer (BPE) + Sampling + KV Cache
   ═══════════════════════════════════════════════════════════════════
   Minimal LLM inference runtime. Weights live in the solver substrate
   (P2 memory fields), not in RAM arrays. The transformer graph uses
   the existing triad_ml.h tensor+autograd for forward pass, and the
   solver for weight storage/retrieval.
   ═══════════════════════════════════════════════════════════════════ */

#ifndef TRIAD_LLM_H
#define TRIAD_LLM_H

#include "triad_ml.h"
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════
   BPE Tokenizer
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    char   *token;      /* UTF-8 string */
    int32_t id;
} TriadVocabEntry;

typedef struct {
    char *left;
    char *right;
    char *merged;       /* left + right concatenated */
    int32_t rank;       /* lower = higher priority */
} TriadMergeRule;

typedef struct {
    const char *key;
    int32_t value;
} TriadTokenHashEntry;

typedef struct {
    char *key;
    int32_t value;
} TriadMergeHashEntry;

typedef struct {
    int32_t          vocab_size;
    TriadVocabEntry *vocab;         /* id -> token string */
    int32_t          nmerges;
    TriadMergeRule  *merges;        /* merge rules sorted by rank */
    int32_t          token_hash_cap;
    TriadTokenHashEntry *token_hash; /* token string -> id */
    int32_t          merge_hash_cap;
    TriadMergeHashEntry *merge_hash; /* left + sep + right -> merge index */
    int32_t          bos_id;        /* beginning of sequence */
    int32_t          eos_id;        /* end of sequence */
    int32_t          pad_id;
} TriadTokenizer;

TriadTokenizer *triad_tokenizer_new(void);
void            triad_tokenizer_free(TriadTokenizer *t);

/* Load vocab + merges from files (sentencepiece/tiktoken compatible) */
int  triad_tokenizer_load_vocab(TriadTokenizer *t, const char *vocab_path);
int  triad_tokenizer_load_merges(TriadTokenizer *t, const char *merges_path);
int  triad_tokenizer_load_added_tokens(TriadTokenizer *t, const char *config_path);

/* Add a single token to vocab */
void triad_tokenizer_add_token(TriadTokenizer *t, int32_t id, const char *token);

/* Add a merge rule */
void triad_tokenizer_add_merge(TriadTokenizer *t, const char *left,
                                const char *right, int32_t rank);

/* Encode text -> token IDs (returns count, fills out_ids) */
int32_t triad_tokenizer_encode(const TriadTokenizer *t, const char *text,
                                int32_t *out_ids, int32_t max_tokens);

/* Decode token IDs -> text (returns allocated string, caller frees) */
char *triad_tokenizer_decode(const TriadTokenizer *t, const int32_t *ids,
                              int32_t n);

/* Decode single token */
const char *triad_tokenizer_id_to_str(const TriadTokenizer *t, int32_t id);

/* ═══════════════════════════════════════════════════════════════════
   Sampling
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    double  temperature;
    int32_t top_k;
    double  top_p;
    uint64_t seed;
} TriadSamplerConfig;

/* Sample next token from logits tensor (vocab_size,) */
int32_t triad_sample_greedy(const double *logits, int32_t vocab_size);
int32_t triad_sample_top_k(const double *logits, int32_t vocab_size,
                            int32_t k, double temperature, uint64_t *rng);
int32_t triad_sample_top_p(const double *logits, int32_t vocab_size,
                            double p, double temperature, uint64_t *rng);
int32_t triad_sample(const double *logits, int32_t vocab_size,
                      const TriadSamplerConfig *cfg);

/* ═══════════════════════════════════════════════════════════════════
   KV Cache (backed by solver memory fields)
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int32_t  max_seq_len;
    int32_t  n_heads;
    int32_t  head_dim;
    int32_t  n_layers;
    int32_t  pos;           /* current position in sequence */

    /* K and V tensors per layer: (max_seq_len, n_heads, head_dim) */
    double **k_cache;       /* k_cache[layer] -> flat array */
    double **v_cache;       /* v_cache[layer] -> flat array */
} TriadKVCache;

TriadKVCache *triad_kv_cache_new(int32_t max_seq_len, int32_t n_heads,
                                  int32_t head_dim, int32_t n_layers);
void          triad_kv_cache_free(TriadKVCache *kv);
void          triad_kv_cache_clear(TriadKVCache *kv);

/* Write K/V for current position at given layer */
void triad_kv_cache_write(TriadKVCache *kv, int32_t layer,
                           const double *k, const double *v);

/* Read K/V up to current position for given layer */
/* Returns pointers into the cache (no copy) */
const double *triad_kv_cache_get_k(const TriadKVCache *kv, int32_t layer);
const double *triad_kv_cache_get_v(const TriadKVCache *kv, int32_t layer);

/* Advance position */
void triad_kv_cache_advance(TriadKVCache *kv);

/* ═══════════════════════════════════════════════════════════════════
   Transformer Ops (RMSNorm, RoPE, Attention, FFN)
   ═══════════════════════════════════════════════════════════════════ */

/* RMS Normalization: out = x / rms(x) * weight */
TriadTensor *triad_rms_norm(TriadTensor *x, TriadTensor *weight, double eps);

/* Rotary Position Embedding: modifies q and k in-place */
void triad_rope(double *q, double *k, int32_t head_dim, int32_t pos,
                double theta_base);

/* Multi-head attention with KV cache */
TriadTensor *triad_mha_cached(TriadTensor *q, TriadTensor *k, TriadTensor *v,
                               TriadKVCache *kv, int32_t layer,
                               int32_t n_heads, int32_t head_dim);

/* SwiGLU FFN: out = down(silu(gate(x)) * up(x)) */
TriadTensor *triad_ffn_swiglu(TriadTensor *x, TriadTensor *gate_w,
                               TriadTensor *up_w, TriadTensor *down_w);

/* ═══════════════════════════════════════════════════════════════════
   Transformer Model (minimal LLaMA-style)
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int32_t dim;            /* model dimension */
    int32_t hidden_dim;     /* FFN intermediate dim */
    int32_t n_layers;
    int32_t n_heads;
    int32_t vocab_size;
    int32_t max_seq_len;
    double  norm_eps;
    double  rope_theta;
} TriadLLMConfig;

typedef struct {
    TriadTensor *wq, *wk, *wv, *wo;    /* attention weights */
    TriadTensor *w_gate, *w_up, *w_down; /* FFN weights */
    TriadTensor *attn_norm, *ffn_norm;   /* RMSNorm weights */
} TriadTransformerLayer;

typedef struct {
    TriadLLMConfig         config;
    TriadTensor           *tok_emb;      /* (vocab_size, dim) */
    TriadTransformerLayer *layers;        /* [n_layers] */
    TriadTensor           *norm;          /* final RMSNorm */
    TriadTensor           *output;        /* output projection (vocab_size, dim) */
    TriadKVCache          *kv_cache;
    TriadTokenizer        *tokenizer;
} TriadLLM;

TriadLLM *triad_llm_new(const TriadLLMConfig *cfg);
void      triad_llm_free(TriadLLM *m);

/* Forward pass for single token at position pos -> logits (vocab_size,) */
TriadTensor *triad_llm_forward(TriadLLM *m, int32_t token_id, int32_t pos);

/* Generate text */
char *triad_llm_generate(TriadLLM *m, const char *prompt, int32_t max_tokens,
                          const TriadSamplerConfig *cfg);

#endif /* TRIAD_LLM_H */
