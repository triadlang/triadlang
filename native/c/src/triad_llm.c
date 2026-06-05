/* ═══════════════════════════════════════════════════════════════════
   TriadLang Native LLM — Tokenizer + Sampling + KV Cache + Transformer
   ═══════════════════════════════════════════════════════════════════ */

#define _POSIX_C_SOURCE 200809L
#include "triad_llm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════
   RNG (xorshift64)
   ═══════════════════════════════ */

static uint64_t _llm_xor(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

static double _llm_randf(uint64_t *s) {
    return (_llm_xor(s) >> 11) * (1.0 / 9007199254740992.0);
}

static uint64_t hash_str(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

static int token_hash_put(TriadTokenizer *t, const char *key, int32_t value) {
    if (!t->token_hash_cap) {
        t->token_hash_cap = 1 << 20;
        t->token_hash = calloc((size_t)t->token_hash_cap, sizeof(TriadTokenHashEntry));
        if (!t->token_hash) return -1;
    }
    uint64_t h = hash_str(key);
    for (int32_t n = 0; n < t->token_hash_cap; n++) {
        int32_t i = (int32_t)((h + (uint64_t)n) & (uint64_t)(t->token_hash_cap - 1));
        if (!t->token_hash[i].key || strcmp(t->token_hash[i].key, key) == 0) {
            t->token_hash[i].key = key;
            t->token_hash[i].value = value;
            return 0;
        }
    }
    return -1;
}

static int32_t token_hash_get(const TriadTokenizer *t, const char *key) {
    if (!t->token_hash || !t->token_hash_cap) return -1;
    uint64_t h = hash_str(key);
    for (int32_t n = 0; n < t->token_hash_cap; n++) {
        int32_t i = (int32_t)((h + (uint64_t)n) & (uint64_t)(t->token_hash_cap - 1));
        if (!t->token_hash[i].key) return -1;
        if (strcmp(t->token_hash[i].key, key) == 0) return t->token_hash[i].value;
    }
    return -1;
}

static char *merge_key_new(const char *left, const char *right) {
    size_t a = strlen(left), b = strlen(right);
    char *key = malloc(a + b + 2);
    if (!key) return NULL;
    memcpy(key, left, a);
    key[a] = '\x01';
    memcpy(key + a + 1, right, b);
    key[a + b + 1] = 0;
    return key;
}

static int merge_hash_put(TriadTokenizer *t, char *key, int32_t value) {
    if (!t->merge_hash_cap) {
        t->merge_hash_cap = 1 << 21;
        t->merge_hash = calloc((size_t)t->merge_hash_cap, sizeof(TriadMergeHashEntry));
        if (!t->merge_hash) return -1;
    }
    uint64_t h = hash_str(key);
    for (int32_t n = 0; n < t->merge_hash_cap; n++) {
        int32_t i = (int32_t)((h + (uint64_t)n) & (uint64_t)(t->merge_hash_cap - 1));
        if (!t->merge_hash[i].key || strcmp(t->merge_hash[i].key, key) == 0) {
            if (t->merge_hash[i].key && t->merge_hash[i].key != key) free(key);
            t->merge_hash[i].key = key;
            t->merge_hash[i].value = value;
            return 0;
        }
    }
    return -1;
}

static int32_t merge_hash_get(const TriadTokenizer *t, const char *left, const char *right) {
    if (!t->merge_hash || !t->merge_hash_cap) return -1;
    char *key = merge_key_new(left, right);
    if (!key) return -1;
    uint64_t h = hash_str(key);
    for (int32_t n = 0; n < t->merge_hash_cap; n++) {
        int32_t i = (int32_t)((h + (uint64_t)n) & (uint64_t)(t->merge_hash_cap - 1));
        if (!t->merge_hash[i].key) {
            free(key);
            return -1;
        }
        if (strcmp(t->merge_hash[i].key, key) == 0) {
            int32_t v = t->merge_hash[i].value;
            free(key);
            return v;
        }
    }
    free(key);
    return -1;
}

/* ═══════════════════════════════
   BPE Tokenizer
   ═══════════════════════════════ */

TriadTokenizer *triad_tokenizer_new(void) {
    TriadTokenizer *t = calloc(1, sizeof(TriadTokenizer));
    t->bos_id = 1;
    t->eos_id = 2;
    t->pad_id = 0;
    return t;
}

void triad_tokenizer_free(TriadTokenizer *t) {
    if (!t) return;
    for (int32_t i = 0; i < t->vocab_size; i++)
        free(t->vocab[i].token);
    free(t->vocab);
    for (int32_t i = 0; i < t->nmerges; i++) {
        free(t->merges[i].left);
        free(t->merges[i].right);
        free(t->merges[i].merged);
    }
    free(t->merges);
    free(t->token_hash);
    if (t->merge_hash) {
        for (int32_t i = 0; i < t->merge_hash_cap; i++) free(t->merge_hash[i].key);
        free(t->merge_hash);
    }
    free(t);
}

void triad_tokenizer_add_token(TriadTokenizer *t, int32_t id, const char *token) {
    /* Grow vocab if needed */
    if (id >= t->vocab_size) {
        int32_t new_size = id + 1;
        t->vocab = realloc(t->vocab, new_size * sizeof(TriadVocabEntry));
        for (int32_t i = t->vocab_size; i < new_size; i++) {
            t->vocab[i].token = NULL;
            t->vocab[i].id = i;
        }
        t->vocab_size = new_size;
    }
    free(t->vocab[id].token);
    t->vocab[id].token = strdup(token);
    t->vocab[id].id = id;
    token_hash_put(t, t->vocab[id].token, id);
}

void triad_tokenizer_add_merge(TriadTokenizer *t, const char *left,
                                const char *right, int32_t rank) {
    t->nmerges++;
    t->merges = realloc(t->merges, t->nmerges * sizeof(TriadMergeRule));
    TriadMergeRule *m = &t->merges[t->nmerges - 1];
    m->left = strdup(left);
    m->right = strdup(right);
    /* merged = left + right */
    size_t ll = strlen(left), lr = strlen(right);
    m->merged = malloc(ll + lr + 1);
    memcpy(m->merged, left, ll);
    memcpy(m->merged + ll, right, lr);
    m->merged[ll + lr] = '\0';
    m->rank = rank;
    char *key = merge_key_new(left, right);
    if (key) merge_hash_put(t, key, t->nmerges - 1);
}

static char *parse_json_string_token(const char **pp) {
    const char *p = *pp;
    if (*p != '"') return NULL;
    p++;
    char *out = malloc(strlen(p) + 1);
    if (!out) return NULL;
    size_t n = 0;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            if (*p == '"' || *p == '\\' || *p == '/') out[n++] = *p++;
            else if (*p == 'n') { out[n++] = '\n'; p++; }
            else if (*p == 'r') { out[n++] = '\r'; p++; }
            else if (*p == 't') { out[n++] = '\t'; p++; }
            else if (*p == 'u' && p[1] && p[2] && p[3] && p[4]) {
                /* Keep unicode escapes as a conservative fallback. Qwen's
                   vocab.json stores byte-unicode tokens directly in UTF-8. */
                out[n++] = '\\';
                out[n++] = *p++;
            } else if (*p) {
                out[n++] = *p++;
            }
        } else {
            out[n++] = *p++;
        }
    }
    out[n] = '\0';
    if (*p == '"') p++;
    *pp = p;
    return out;
}

static void utf8_append(char *out, size_t *n, int cp) {
    if (cp < 0x80) {
        out[(*n)++] = (char)cp;
    } else if (cp < 0x800) {
        out[(*n)++] = (char)(0xC0 | (cp >> 6));
        out[(*n)++] = (char)(0x80 | (cp & 0x3F));
    } else {
        out[(*n)++] = (char)(0xE0 | (cp >> 12));
        out[(*n)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[(*n)++] = (char)(0x80 | (cp & 0x3F));
    }
}

static void byte_to_gpt2_symbol(unsigned char b, char out[8]) {
    size_t n = 0;
    if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE)) {
        utf8_append(out, &n, b);
    } else {
        int extra = 0;
        for (int x = 0; x < b; x++) {
            if (!((x >= '!' && x <= '~') || (x >= 0xA1 && x <= 0xAC) || (x >= 0xAE)))
                extra++;
        }
        utf8_append(out, &n, 256 + extra);
    }
    out[n] = 0;
}

int triad_tokenizer_load_vocab(TriadTokenizer *t, const char *vocab_path) {
    FILE *fp = fopen(vocab_path, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) { fclose(fp); return -1; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return -1; }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        free(buf); fclose(fp); return -1;
    }
    fclose(fp);
    buf[sz] = '\0';

    const char *p = buf;
    while (*p) {
        while (*p && *p != '"') p++;
        if (!*p) break;
        char *tok = parse_json_string_token(&p);
        if (!tok) break;
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
        if (*p != ':') { free(tok); continue; }
        p++;
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
        char *end = NULL;
        long id = strtol(p, &end, 10);
        if (end != p && id >= 0) triad_tokenizer_add_token(t, (int32_t)id, tok);
        free(tok);
        p = end ? end : p;
    }
    free(buf);
    return 0;
}

int triad_tokenizer_load_merges(TriadTokenizer *t, const char *merges_path) {
    FILE *fp = fopen(merges_path, "r");
    if (!fp) return -1;
    char line[4096];
    int32_t rank = 0;
    while (fgets(line, sizeof(line), fp)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        if (n == 0 || line[0] == '#') continue;
        char *sp = strchr(line, ' ');
        if (!sp) continue;
        *sp = '\0';
        triad_tokenizer_add_merge(t, line, sp + 1, rank++);
    }
    fclose(fp);
    return 0;
}

int triad_tokenizer_load_added_tokens(TriadTokenizer *t, const char *config_path) {
    FILE *fp = fopen(config_path, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) { fclose(fp); return -1; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return -1; }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        free(buf); fclose(fp); return -1;
    }
    fclose(fp);
    buf[sz] = '\0';

    const char *p = strstr(buf, "\"added_tokens_decoder\"");
    if (!p) {
        free(buf);
        return -1;
    }
    p = strchr(p, '{');
    if (!p) {
        free(buf);
        return -1;
    }
    p++;

    int added = 0;
    while (*p) {
        while (*p && *p != '"' && *p != '}') p++;
        if (*p == '}') break;
        char *id_str = parse_json_string_token(&p);
        if (!id_str) break;
        char *end = NULL;
        long id = strtol(id_str, &end, 10);
        free(id_str);
        if (!end || *end != '\0' || id < 0) continue;

        const char *entry_end = strchr(p, '}');
        const char *content_key = strstr(p, "\"content\"");
        if (!entry_end || !content_key || content_key > entry_end) {
            if (entry_end) p = entry_end + 1;
            continue;
        }
        p = strchr(content_key, ':');
        if (!p || p > entry_end) {
            p = entry_end + 1;
            continue;
        }
        p++;
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
        char *content = parse_json_string_token(&p);
        if (content) {
            triad_tokenizer_add_token(t, (int32_t)id, content);
            added++;
            free(content);
        }
        if (entry_end && p < entry_end) p = entry_end + 1;
    }

    free(buf);
    return added > 0 ? 0 : -1;
}

/* BPE encode: bytes -> GPT-2 byte-unicode symbols -> merges */
int32_t triad_tokenizer_encode(const TriadTokenizer *t, const char *text,
                                int32_t *out_ids, int32_t max_tokens) {
    int len = (int)strlen(text);
    if (len == 0) return 0;

    typedef struct { char *str; int32_t id; } TokPiece;
    TokPiece *pieces = malloc((len + 1) * sizeof(TokPiece));
    int npieces = 0;

    for (int i = 0; i < len; i++) {
        char sym[8];
        byte_to_gpt2_symbol((unsigned char)text[i], sym);
        pieces[npieces].str = strdup(sym);
        pieces[npieces].id = token_hash_get(t, sym);
        npieces++;
    }

    /* Apply BPE merges iteratively */
    int changed = 1;
    while (changed) {
        changed = 0;
        int best_rank = INT32_MAX;
        int best_pos = -1;
        int best_merge = -1;

        /* Find highest priority merge */
        for (int i = 0; i < npieces - 1; i++) {
            int32_t m = merge_hash_get(t, pieces[i].str, pieces[i + 1].str);
            if (m >= 0 && t->merges[m].rank < best_rank) {
                best_rank = t->merges[m].rank;
                best_pos = i;
                best_merge = m;
            }
        }

        if (best_pos >= 0) {
            /* Apply merge at best_pos */
            free(pieces[best_pos].str);
            pieces[best_pos].str = strdup(t->merges[best_merge].merged);
            pieces[best_pos].id = token_hash_get(t, pieces[best_pos].str);
            /* Remove piece at best_pos+1 */
            free(pieces[best_pos + 1].str);
            for (int i = best_pos + 1; i < npieces - 1; i++)
                pieces[i] = pieces[i + 1];
            npieces--;
            changed = 1;
        }
    }

    /* Emit token IDs (free every piece regardless of the output cap) */
    int32_t count = 0;
    for (int i = 0; i < npieces; i++) {
        if (pieces[i].id >= 0 && count < max_tokens) {
            out_ids[count++] = pieces[i].id;
        }
        free(pieces[i].str);
    }
    free(pieces);
    return count;
}

char *triad_tokenizer_decode(const TriadTokenizer *t, const int32_t *ids, int32_t n) {
    /* Concatenate token strings */
    size_t total = 0;
    for (int32_t i = 0; i < n; i++) {
        if (ids[i] >= 0 && ids[i] < t->vocab_size && t->vocab[ids[i]].token)
            total += strlen(t->vocab[ids[i]].token);
    }
    char *out = malloc(total + 1);
    out[0] = '\0';
    for (int32_t i = 0; i < n; i++) {
        if (ids[i] >= 0 && ids[i] < t->vocab_size && t->vocab[ids[i]].token)
            strcat(out, t->vocab[ids[i]].token);
    }
    return out;
}

const char *triad_tokenizer_id_to_str(const TriadTokenizer *t, int32_t id) {
    if (id >= 0 && id < t->vocab_size && t->vocab[id].token)
        return t->vocab[id].token;
    return "<unk>";
}

/* ═══════════════════════════════
   Sampling
   ═══════════════════════════════ */

int32_t triad_sample_greedy(const double *logits, int32_t vocab_size) {
    int32_t best = 0;
    double best_val = logits[0];
    for (int32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

int32_t triad_sample_top_k(const double *logits, int32_t vocab_size,
                            int32_t k, double temperature, uint64_t *rng) {
    if (k <= 0 || k > vocab_size) k = vocab_size;

    /* Find top-k indices (partial sort) */
    typedef struct { double val; int32_t idx; } LP;
    LP *pairs = malloc(vocab_size * sizeof(LP));
    for (int32_t i = 0; i < vocab_size; i++) {
        pairs[i].val = logits[i] / (temperature > 0 ? temperature : 1.0);
        pairs[i].idx = i;
    }
    /* Partial sort: bubble top-k to front */
    for (int32_t i = 0; i < k; i++) {
        for (int32_t j = i + 1; j < vocab_size; j++) {
            if (pairs[j].val > pairs[i].val) {
                LP tmp = pairs[i]; pairs[i] = pairs[j]; pairs[j] = tmp;
            }
        }
    }

    /* Softmax over top-k */
    double max_val = pairs[0].val;
    double sum = 0;
    for (int32_t i = 0; i < k; i++) {
        pairs[i].val = exp(pairs[i].val - max_val);
        sum += pairs[i].val;
    }

    /* Sample */
    double r = _llm_randf(rng) * sum;
    double cum = 0;
    int32_t result = pairs[0].idx;
    for (int32_t i = 0; i < k; i++) {
        cum += pairs[i].val;
        if (cum >= r) { result = pairs[i].idx; break; }
    }
    free(pairs);
    return result;
}

int32_t triad_sample_top_p(const double *logits, int32_t vocab_size,
                            double p, double temperature, uint64_t *rng) {
    typedef struct { double val; int32_t idx; } LP;
    LP *pairs = malloc(vocab_size * sizeof(LP));
    double max_val = -1e308;
    for (int32_t i = 0; i < vocab_size; i++) {
        pairs[i].val = logits[i] / (temperature > 0 ? temperature : 1.0);
        pairs[i].idx = i;
        if (pairs[i].val > max_val) max_val = pairs[i].val;
    }

    /* Softmax */
    double sum = 0;
    for (int32_t i = 0; i < vocab_size; i++) {
        pairs[i].val = exp(pairs[i].val - max_val);
        sum += pairs[i].val;
    }
    for (int32_t i = 0; i < vocab_size; i++)
        pairs[i].val /= sum;

    /* Sort descending by probability */
    for (int32_t i = 0; i < vocab_size - 1; i++)
        for (int32_t j = i + 1; j < vocab_size; j++)
            if (pairs[j].val > pairs[i].val) {
                LP tmp = pairs[i]; pairs[i] = pairs[j]; pairs[j] = tmp;
            }

    /* Accumulate until p */
    double cum = 0;
    int32_t cutoff = vocab_size;
    for (int32_t i = 0; i < vocab_size; i++) {
        cum += pairs[i].val;
        if (cum >= p) { cutoff = i + 1; break; }
    }

    /* Renormalize and sample */
    sum = 0;
    for (int32_t i = 0; i < cutoff; i++) sum += pairs[i].val;
    double r = _llm_randf(rng) * sum;
    cum = 0;
    int32_t result = pairs[0].idx;
    for (int32_t i = 0; i < cutoff; i++) {
        cum += pairs[i].val;
        if (cum >= r) { result = pairs[i].idx; break; }
    }
    free(pairs);
    return result;
}

int32_t triad_sample(const double *logits, int32_t vocab_size,
                      const TriadSamplerConfig *cfg) {
    uint64_t rng = cfg->seed;
    if (cfg->temperature <= 0)
        return triad_sample_greedy(logits, vocab_size);
    if (cfg->top_p > 0 && cfg->top_p < 1.0)
        return triad_sample_top_p(logits, vocab_size, cfg->top_p, cfg->temperature, &rng);
    if (cfg->top_k > 0)
        return triad_sample_top_k(logits, vocab_size, cfg->top_k, cfg->temperature, &rng);
    return triad_sample_top_k(logits, vocab_size, vocab_size, cfg->temperature, &rng);
}

/* ═══════════════════════════════
   KV Cache
   ═══════════════════════════════ */

TriadKVCache *triad_kv_cache_new(int32_t max_seq_len, int32_t n_heads,
                                  int32_t head_dim, int32_t n_layers) {
    TriadKVCache *kv = calloc(1, sizeof(TriadKVCache));
    kv->max_seq_len = max_seq_len;
    kv->n_heads = n_heads;
    kv->head_dim = head_dim;
    kv->n_layers = n_layers;
    kv->pos = 0;

    int64_t layer_size = (int64_t)max_seq_len * n_heads * head_dim;
    kv->k_cache = malloc(n_layers * sizeof(double*));
    kv->v_cache = malloc(n_layers * sizeof(double*));
    for (int32_t l = 0; l < n_layers; l++) {
        kv->k_cache[l] = calloc(layer_size, sizeof(double));
        kv->v_cache[l] = calloc(layer_size, sizeof(double));
    }
    return kv;
}

void triad_kv_cache_free(TriadKVCache *kv) {
    if (!kv) return;
    for (int32_t l = 0; l < kv->n_layers; l++) {
        free(kv->k_cache[l]);
        free(kv->v_cache[l]);
    }
    free(kv->k_cache);
    free(kv->v_cache);
    free(kv);
}

void triad_kv_cache_clear(TriadKVCache *kv) {
    kv->pos = 0;
    int64_t layer_size = (int64_t)kv->max_seq_len * kv->n_heads * kv->head_dim;
    for (int32_t l = 0; l < kv->n_layers; l++) {
        memset(kv->k_cache[l], 0, layer_size * sizeof(double));
        memset(kv->v_cache[l], 0, layer_size * sizeof(double));
    }
}

void triad_kv_cache_write(TriadKVCache *kv, int32_t layer,
                           const double *k, const double *v) {
    int64_t off = (int64_t)kv->pos * kv->n_heads * kv->head_dim;
    int64_t sz = (int64_t)kv->n_heads * kv->head_dim;
    memcpy(kv->k_cache[layer] + off, k, sz * sizeof(double));
    memcpy(kv->v_cache[layer] + off, v, sz * sizeof(double));
}

const double *triad_kv_cache_get_k(const TriadKVCache *kv, int32_t layer) {
    return kv->k_cache[layer];
}

const double *triad_kv_cache_get_v(const TriadKVCache *kv, int32_t layer) {
    return kv->v_cache[layer];
}

void triad_kv_cache_advance(TriadKVCache *kv) {
    kv->pos++;
    if (kv->pos >= kv->max_seq_len) {
        /* Ring buffer: wrap around */
        kv->pos = 0;
    }
}

/* ═══════════════════════════════
   Transformer Ops
   ═══════════════════════════════ */

/* RMS Norm */
TriadTensor *triad_rms_norm(TriadTensor *x, TriadTensor *weight, double eps) {
    int64_t dim = x->size;
    double ss = 0;
    for (int64_t i = 0; i < dim; i++) ss += x->data[i] * x->data[i];
    ss = 1.0 / sqrt(ss / (double)dim + eps);

    int32_t shape[] = {(int32_t)dim};
    TriadTensor *out = triad_tensor_new(1, shape, 0);
    for (int64_t i = 0; i < dim; i++)
        out->data[i] = x->data[i] * ss * weight->data[i];
    return out;
}

/* RoPE: apply rotary embeddings to q and k */
void triad_rope(double *q, double *k, int32_t head_dim, int32_t pos,
                double theta_base) {
    for (int32_t i = 0; i < head_dim; i += 2) {
        double freq = 1.0 / pow(theta_base, (double)i / (double)head_dim);
        double angle = (double)pos * freq;
        double cos_a = cos(angle), sin_a = sin(angle);

        /* Rotate q */
        double q0 = q[i], q1 = q[i + 1];
        q[i]     = q0 * cos_a - q1 * sin_a;
        q[i + 1] = q0 * sin_a + q1 * cos_a;

        /* Rotate k */
        double k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * cos_a - k1 * sin_a;
        k[i + 1] = k0 * sin_a + k1 * cos_a;
    }
}

/* Single-head scaled dot-product attention with causal mask */
static void _sdpa(const double *q, const double *k_cache, const double *v_cache,
                   double *out, int32_t head_dim, int32_t seq_len,
                   int32_t n_heads, int32_t head_idx) {
    double scale = 1.0 / sqrt((double)head_dim);
    double *scores = malloc((seq_len + 1) * sizeof(double));

    /* QK^T */
    double max_score = -1e308;
    for (int32_t t = 0; t <= seq_len; t++) {
        double dot = 0;
        for (int32_t d = 0; d < head_dim; d++) {
            int64_t idx = (int64_t)t * n_heads * head_dim + head_idx * head_dim + d;
            dot += q[d] * k_cache[idx];
        }
        scores[t] = dot * scale;
        if (scores[t] > max_score) max_score = scores[t];
    }

    /* Softmax */
    double sum = 0;
    for (int32_t t = 0; t <= seq_len; t++) {
        scores[t] = exp(scores[t] - max_score);
        sum += scores[t];
    }
    for (int32_t t = 0; t <= seq_len; t++)
        scores[t] /= sum;

    /* Weighted sum of V */
    memset(out, 0, head_dim * sizeof(double));
    for (int32_t t = 0; t <= seq_len; t++) {
        for (int32_t d = 0; d < head_dim; d++) {
            int64_t idx = (int64_t)t * n_heads * head_dim + head_idx * head_dim + d;
            out[d] += scores[t] * v_cache[idx];
        }
    }
    free(scores);
}

/* Multi-head attention with KV cache (inference, single token) */
TriadTensor *triad_mha_cached(TriadTensor *q_proj, TriadTensor *k_proj, TriadTensor *v_proj,
                               TriadKVCache *kv, int32_t layer,
                               int32_t n_heads, int32_t head_dim) {
    /* Write current K/V to cache */
    triad_kv_cache_write(kv, layer, k_proj->data, v_proj->data);

    /* Per-head attention */
    int32_t dim = n_heads * head_dim;
    int32_t shape[] = {dim};
    TriadTensor *out = triad_tensor_new(1, shape, 0);

    const double *k_cache = triad_kv_cache_get_k(kv, layer);
    const double *v_cache = triad_kv_cache_get_v(kv, layer);

    for (int32_t h = 0; h < n_heads; h++) {
        _sdpa(q_proj->data + h * head_dim,
              k_cache, v_cache,
              out->data + h * head_dim,
              head_dim, kv->pos, n_heads, h);
    }
    return out;
}

/* SwiGLU FFN */
TriadTensor *triad_ffn_swiglu(TriadTensor *x, TriadTensor *gate_w,
                               TriadTensor *up_w, TriadTensor *down_w) {
    /* gate = x @ gate_w^T */
    /* up   = x @ up_w^T */
    int32_t dim = x->shape[0];
    int32_t hidden = gate_w->shape[0];

    int32_t hs[] = {hidden};
    TriadTensor *gate = triad_tensor_new(1, hs, 0);
    TriadTensor *up   = triad_tensor_new(1, hs, 0);

    for (int32_t j = 0; j < hidden; j++) {
        double gs = 0, us = 0;
        for (int32_t k = 0; k < dim; k++) {
            gs += x->data[k] * gate_w->data[j * dim + k];
            us += x->data[k] * up_w->data[j * dim + k];
        }
        /* SiLU activation on gate */
        double silu = gs / (1.0 + exp(-gs));
        gate->data[j] = silu * us;
    }

    /* down = hidden @ down_w^T */
    int32_t ds[] = {dim};
    TriadTensor *out = triad_tensor_new(1, ds, 0);
    for (int32_t j = 0; j < dim; j++) {
        double s = 0;
        for (int32_t k = 0; k < hidden; k++)
            s += gate->data[k] * down_w->data[j * hidden + k];
        out->data[j] = s;
    }

    triad_tensor_free(gate);
    triad_tensor_free(up);
    return out;
}

/* ═══════════════════════════════
   Transformer Model
   ═══════════════════════════════ */

TriadLLM *triad_llm_new(const TriadLLMConfig *cfg) {
    TriadLLM *m = calloc(1, sizeof(TriadLLM));
    m->config = *cfg;

    int32_t dim = cfg->dim;
    int32_t hidden = cfg->hidden_dim;
    int32_t vocab = cfg->vocab_size;
    int32_t head_dim = dim / cfg->n_heads;

    /* Embedding */
    int32_t es[] = {vocab, dim};
    m->tok_emb = triad_tensor_new(2, es, 0);

    /* Layers */
    m->layers = calloc(cfg->n_layers, sizeof(TriadTransformerLayer));
    for (int32_t l = 0; l < cfg->n_layers; l++) {
        TriadTransformerLayer *lay = &m->layers[l];
        int32_t ws[] = {dim, dim};
        lay->wq = triad_tensor_new(2, ws, 0);
        lay->wk = triad_tensor_new(2, ws, 0);
        lay->wv = triad_tensor_new(2, ws, 0);
        lay->wo = triad_tensor_new(2, ws, 0);

        int32_t fs[] = {hidden, dim};
        lay->w_gate = triad_tensor_new(2, fs, 0);
        lay->w_up   = triad_tensor_new(2, fs, 0);
        int32_t ds[] = {dim, hidden};
        lay->w_down  = triad_tensor_new(2, ds, 0);

        int32_t ns[] = {dim};
        lay->attn_norm = triad_tensor_ones(1, ns, 0);
        lay->ffn_norm  = triad_tensor_ones(1, ns, 0);
    }

    /* Final norm + output */
    int32_t ns[] = {dim};
    m->norm = triad_tensor_ones(1, ns, 0);
    int32_t os[] = {vocab, dim};
    m->output = triad_tensor_new(2, os, 0);

    /* KV cache */
    m->kv_cache = triad_kv_cache_new(cfg->max_seq_len, cfg->n_heads, head_dim, cfg->n_layers);

    return m;
}

void triad_llm_free(TriadLLM *m) {
    if (!m) return;
    triad_tensor_free(m->tok_emb);
    for (int32_t l = 0; l < m->config.n_layers; l++) {
        TriadTransformerLayer *lay = &m->layers[l];
        triad_tensor_free(lay->wq);
        triad_tensor_free(lay->wk);
        triad_tensor_free(lay->wv);
        triad_tensor_free(lay->wo);
        triad_tensor_free(lay->w_gate);
        triad_tensor_free(lay->w_up);
        triad_tensor_free(lay->w_down);
        triad_tensor_free(lay->attn_norm);
        triad_tensor_free(lay->ffn_norm);
    }
    free(m->layers);
    triad_tensor_free(m->norm);
    triad_tensor_free(m->output);
    triad_kv_cache_free(m->kv_cache);
    triad_tokenizer_free(m->tokenizer);
    free(m);
}

/* Forward: single token -> logits */
TriadTensor *triad_llm_forward(TriadLLM *m, int32_t token_id, int32_t pos) {
    int32_t dim = m->config.dim;
    int32_t n_heads = m->config.n_heads;
    int32_t head_dim = dim / n_heads;

    /* Token embedding */
    int32_t xs[] = {dim};
    TriadTensor *x = triad_tensor_new(1, xs, 0);
    memcpy(x->data, m->tok_emb->data + (int64_t)token_id * dim, dim * sizeof(double));

    /* Transformer layers */
    for (int32_t l = 0; l < m->config.n_layers; l++) {
        TriadTransformerLayer *lay = &m->layers[l];

        /* Pre-attention RMSNorm */
        TriadTensor *xn = triad_rms_norm(x, lay->attn_norm, m->config.norm_eps);

        /* QKV projections */
        TriadTensor *q = triad_tensor_new(1, xs, 0);
        TriadTensor *k = triad_tensor_new(1, xs, 0);
        TriadTensor *v = triad_tensor_new(1, xs, 0);
        for (int32_t i = 0; i < dim; i++) {
            double qs = 0, ks = 0, vs = 0;
            for (int32_t j = 0; j < dim; j++) {
                qs += xn->data[j] * lay->wq->data[i * dim + j];
                ks += xn->data[j] * lay->wk->data[i * dim + j];
                vs += xn->data[j] * lay->wv->data[i * dim + j];
            }
            q->data[i] = qs;
            k->data[i] = ks;
            v->data[i] = vs;
        }

        /* RoPE per head */
        for (int32_t h = 0; h < n_heads; h++) {
            triad_rope(q->data + h * head_dim, k->data + h * head_dim,
                       head_dim, pos, m->config.rope_theta);
        }

        /* Multi-head attention with KV cache */
        TriadTensor *attn_out = triad_mha_cached(q, k, v, m->kv_cache, l,
                                                  n_heads, head_dim);

        /* Output projection */
        TriadTensor *proj = triad_tensor_new(1, xs, 0);
        for (int32_t i = 0; i < dim; i++) {
            double s = 0;
            for (int32_t j = 0; j < dim; j++)
                s += attn_out->data[j] * lay->wo->data[i * dim + j];
            proj->data[i] = s;
        }

        /* Residual */
        for (int32_t i = 0; i < dim; i++)
            x->data[i] += proj->data[i];

        triad_tensor_free(xn);
        triad_tensor_free(q);
        triad_tensor_free(k);
        triad_tensor_free(v);
        triad_tensor_free(attn_out);
        triad_tensor_free(proj);

        /* Pre-FFN RMSNorm */
        xn = triad_rms_norm(x, lay->ffn_norm, m->config.norm_eps);

        /* SwiGLU FFN */
        TriadTensor *ffn_out = triad_ffn_swiglu(xn, lay->w_gate, lay->w_up, lay->w_down);

        /* Residual */
        for (int32_t i = 0; i < dim; i++)
            x->data[i] += ffn_out->data[i];

        triad_tensor_free(xn);
        triad_tensor_free(ffn_out);
    }

    /* Final RMSNorm */
    TriadTensor *xn = triad_rms_norm(x, m->norm, m->config.norm_eps);

    /* Output projection -> logits */
    int32_t ls[] = {m->config.vocab_size};
    TriadTensor *logits = triad_tensor_new(1, ls, 0);
    for (int32_t i = 0; i < m->config.vocab_size; i++) {
        double s = 0;
        for (int32_t j = 0; j < dim; j++)
            s += xn->data[j] * m->output->data[i * dim + j];
        logits->data[i] = s;
    }

    /* Advance KV cache */
    triad_kv_cache_advance(m->kv_cache);

    triad_tensor_free(x);
    triad_tensor_free(xn);
    return logits;
}

/* Generate text from prompt */
char *triad_llm_generate(TriadLLM *m, const char *prompt, int32_t max_tokens,
                          const TriadSamplerConfig *cfg) {
    if (!m->tokenizer) return strdup("");

    /* Encode prompt */
    int32_t prompt_ids[4096];
    int32_t prompt_len = triad_tokenizer_encode(m->tokenizer, prompt, prompt_ids, 4096);

    /* Output buffer */
    int32_t *out_ids = malloc((prompt_len + max_tokens) * sizeof(int32_t));
    memcpy(out_ids, prompt_ids, prompt_len * sizeof(int32_t));
    int32_t total = prompt_len;

    /* Clear KV cache */
    triad_kv_cache_clear(m->kv_cache);

    /* Process prompt (prefill) */
    for (int32_t i = 0; i < prompt_len; i++) {
        TriadTensor *logits = triad_llm_forward(m, prompt_ids[i], i);
        if (i == prompt_len - 1) {
            /* Sample from last token's logits */
            int32_t next = triad_sample(logits->data, m->config.vocab_size, cfg);
            out_ids[total++] = next;
        }
        triad_tensor_free(logits);
    }

    /* Autoregressive generation */
    for (int32_t i = 0; i < max_tokens - 1 && total < prompt_len + max_tokens; i++) {
        int32_t pos = prompt_len + i;
        TriadTensor *logits = triad_llm_forward(m, out_ids[total - 1], pos);
        int32_t next = triad_sample(logits->data, m->config.vocab_size, cfg);
        triad_tensor_free(logits);

        if (next == m->tokenizer->eos_id) break;
        out_ids[total++] = next;
    }

    /* Decode output (skip prompt) */
    char *result = triad_tokenizer_decode(m->tokenizer, out_ids + prompt_len,
                                           total - prompt_len);
    free(out_ids);
    return result;
}
