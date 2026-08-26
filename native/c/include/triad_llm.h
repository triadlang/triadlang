#ifndef TRIAD_LLM_H
#define TRIAD_LLM_H

#include "triad_ml.h"
#include <stdint.h>

typedef struct {
    char   *token;
    int32_t id;
} TriadVocabEntry;

typedef struct {
    char *left;
    char *right;
    char *merged;
    int32_t rank;
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
    TriadVocabEntry *vocab;
    int32_t          nmerges;
    TriadMergeRule  *merges;
    int32_t          token_hash_cap;
    TriadTokenHashEntry *token_hash;
    int32_t          merge_hash_cap;
    TriadMergeHashEntry *merge_hash;
    int32_t          bos_id;
    int32_t          eos_id;
    int32_t          pad_id;
} TriadTokenizer;

TriadTokenizer *triad_tokenizer_new(void);
void            triad_tokenizer_free(TriadTokenizer *t);

int  triad_tokenizer_load_vocab(TriadTokenizer *t, const char *vocab_path);
int  triad_tokenizer_load_merges(TriadTokenizer *t, const char *merges_path);
int  triad_tokenizer_load_added_tokens(TriadTokenizer *t, const char *config_path);

void triad_tokenizer_add_token(TriadTokenizer *t, int32_t id, const char *token);

void triad_tokenizer_add_merge(TriadTokenizer *t, const char *left,
                                const char *right, int32_t rank);

int32_t triad_tokenizer_encode(const TriadTokenizer *t, const char *text,
                                int32_t *out_ids, int32_t max_tokens);

char *triad_tokenizer_decode(const TriadTokenizer *t, const int32_t *ids,
                              int32_t n);

const char *triad_tokenizer_id_to_str(const TriadTokenizer *t, int32_t id);

typedef struct {
    double  temperature;
    int32_t top_k;
    double  top_p;
    uint64_t seed;
} TriadSamplerConfig;

int32_t triad_sample_greedy(const double *logits, int32_t vocab_size);
int32_t triad_sample_top_k(const double *logits, int32_t vocab_size,
                            int32_t k, double temperature, uint64_t *rng);
int32_t triad_sample_top_p(const double *logits, int32_t vocab_size,
                            double p, double temperature, uint64_t *rng);
int32_t triad_sample(const double *logits, int32_t vocab_size,
                      const TriadSamplerConfig *cfg);

typedef struct {
    int32_t  max_seq_len;
    int32_t  n_heads;
    int32_t  head_dim;
    int32_t  n_layers;
    int32_t  pos;

    double **k_cache;
    double **v_cache;
} TriadKVCache;

TriadKVCache *triad_kv_cache_new(int32_t max_seq_len, int32_t n_heads,
                                  int32_t head_dim, int32_t n_layers);
void          triad_kv_cache_free(TriadKVCache *kv);
void          triad_kv_cache_clear(TriadKVCache *kv);

void triad_kv_cache_write(TriadKVCache *kv, int32_t layer,
                           const double *k, const double *v);

const double *triad_kv_cache_get_k(const TriadKVCache *kv, int32_t layer);
const double *triad_kv_cache_get_v(const TriadKVCache *kv, int32_t layer);

void triad_kv_cache_advance(TriadKVCache *kv);

TriadTensor *triad_rms_norm(TriadTensor *x, TriadTensor *weight, double eps);

void triad_rope(double *q, double *k, int32_t head_dim, int32_t pos,
                double theta_base);

TriadTensor *triad_mha_cached(TriadTensor *q, TriadTensor *k, TriadTensor *v,
                               TriadKVCache *kv, int32_t layer,
                               int32_t n_heads, int32_t head_dim);

TriadTensor *triad_ffn_swiglu(TriadTensor *x, TriadTensor *gate_w,
                               TriadTensor *up_w, TriadTensor *down_w);

typedef struct {
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t vocab_size;
    int32_t max_seq_len;
    double  norm_eps;
    double  rope_theta;
} TriadLLMConfig;

typedef struct {
    TriadTensor *wq, *wk, *wv, *wo;
    TriadTensor *w_gate, *w_up, *w_down;
    TriadTensor *attn_norm, *ffn_norm;
} TriadTransformerLayer;

typedef struct {
    TriadLLMConfig         config;
    TriadTensor           *tok_emb;
    TriadTransformerLayer *layers;
    TriadTensor           *norm;
    TriadTensor           *output;
    TriadKVCache          *kv_cache;
    TriadTokenizer        *tokenizer;
} TriadLLM;

TriadLLM *triad_llm_new(const TriadLLMConfig *cfg);
void      triad_llm_free(TriadLLM *m);

TriadTensor *triad_llm_forward(TriadLLM *m, int32_t token_id, int32_t pos);

char *triad_llm_generate(TriadLLM *m, const char *prompt, int32_t max_tokens,
                          const TriadSamplerConfig *cfg);

#endif
