#define _POSIX_C_SOURCE 200809L
#include "triad_chat_backend.h"
#include "triad_gguf.h"
#include "triad_llm.h"
#include "triad_model_index.h"
#include "triad_rt.h"
#include "triad_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sys/stat.h>

#define GRID_N    256
#define GRID_L    32.0
#define DEFAULT_CANDIDATES 0

typedef struct {
    const GGUFTensorInfo *attn_norm;
    const GGUFTensorInfo *attn_qkv;
    const GGUFTensorInfo *attn_gate;
    const GGUFTensorInfo *ssm_alpha;
    const GGUFTensorInfo *ssm_beta;
    const GGUFTensorInfo *ssm_conv;
    const GGUFTensorInfo *ssm_out;
    const GGUFTensorInfo *ssm_norm;
    const GGUFTensorInfo *ffn_norm;
    const GGUFTensorInfo *ffn_gate;
    const GGUFTensorInfo *ffn_up;
    const GGUFTensorInfo *ffn_down;
} TriadGGUFLayer;

typedef struct {
    GGUFFile *gguf;
    const GGUFTensorInfo *tok_emb;
    const GGUFTensorInfo *out_w;
    TriadGGUFLayer *layers;
    int32_t n_layers;
    int32_t dim;
    int32_t vocab_size;
    double *emb_buf;
    double *out_buf;
    double *latent_buf;
    double *layer_buf;
    TriadCplx *field_buf;
} TriadGGUFSubstrate;

typedef struct {
    TriadModelIndex index;
    int32_t dim;
    int32_t vocab_size;
    int32_t n_layers;
    double *emb_buf;
    double *out_buf;
    double *latent_buf;
    double *layer_buf;
    int64_t layer_cap;
    TriadCplx *field_buf;
} TriadSafeSubstrate;

static void encode_token_spectral(int32_t token_id,
                                  TriadCplx *psi, int32_t N, double L) {
    double dx = L / N;
    double k_star = (double)token_id * (TRIAD_PI * 2.0 / L);
    for (int32_t i = 0; i < N; i++) {
        double x = -L/2 + (double)i * dx;
        double env = exp(-x*x / (L*L/8));
        double val = env * (1.0 + 0.9 * cos(k_star * x));
        psi[i] = (TriadCplx){ val, 0.0 };
    }
    double norm = 0;
    for (int32_t i = 0; i < N; i++)
        norm += psi[i].re * psi[i].re;
    norm = sqrt(norm * dx);
    for (int32_t i = 0; i < N; i++)
        psi[i].re /= norm;
}

static void vector_to_psi(const double *v, int32_t dim, TriadCplx *psi,
                          int32_t N, double L) {
    double dx = L / N;
    double norm = 0.0;
    for (int32_t i = 0; i < N; i++) {
        int64_t start = (int64_t)i * dim / N;
        int64_t end = (int64_t)(i + 1) * dim / N;
        if (end <= start) end = start + 1;
        double s = 0.0;
        for (int64_t j = start; j < end && j < dim; j++) s += v[j];
        double val = s / (double)(end - start);
        psi[i] = (TriadCplx){ val, 0.0 };
        norm += val * val;
    }
    norm = sqrt(norm * dx);
    if (norm < 1e-30) norm = 1.0;
    for (int32_t i = 0; i < N; i++) psi[i].re /= norm;
}

static int encode_token_from_weights(TriadGGUFSubstrate *rt, int32_t token_id,
                                     TriadCplx *psi, int32_t N, double L) {
    if (!rt || token_id < 0 || token_id >= rt->vocab_size) return -1;
    if (gguf_dequantize_row(rt->gguf, rt->tok_emb, token_id,
                            rt->emb_buf, rt->dim) != 0) {
        return -1;
    }
    vector_to_psi(rt->emb_buf, rt->dim, psi, N, L);
    return 0;
}

static int encode_token_from_safetensors(TriadSafeSubstrate *rt, int32_t token_id,
                                         TriadCplx *psi, int32_t N, double L) {
    if (!rt || token_id < 0 || token_id >= rt->vocab_size) return -1;
    if (triad_model_read_row_f64(&rt->index, "model.language_model.embed_tokens.weight",
                                 token_id, rt->emb_buf, rt->dim) < 0) {
        return -1;
    }
    vector_to_psi(rt->emb_buf, rt->dim, psi, N, L);
    return 0;
}

static void psi_to_latent(const TriadCplx *psi, int32_t N,
                          double *latent, int32_t dim) {
    for (int32_t j = 0; j < dim; j++) {
        int32_t i = (int64_t)j * N / dim;
        double phase = 2.0 * TRIAD_PI * (double)(j % N) / (double)N;
        latent[j] = psi[i].re * cos(phase) - psi[i].im * sin(phase);
    }
}

static double dot_vec(const double *a, const double *b, int32_t n) {
    double s = 0.0;
    for (int32_t i = 0; i < n; i++) s += a[i] * b[i];
    return s / sqrt((double)n);
}

static int add_candidate_id(int32_t *ids, int32_t *nids, int32_t max_ids,
                            int32_t vocab_size, int32_t id) {
    if (id < 0 || id >= vocab_size || *nids >= max_ids) return 0;
    for (int32_t i = 0; i < *nids; i++)
        if (ids[i] == id) return 0;
    ids[(*nids)++] = id;
    return 1;
}

static void add_encoded_candidates(const TriadTokenizer *tok, int32_t *ids,
                                   int32_t *nids, int32_t max_ids,
                                   int32_t vocab_size, const char *text) {
    int32_t tmp[32];
    int32_t n = triad_tokenizer_encode(tok, text, tmp, 32);
    for (int32_t i = 0; i < n; i++)
        add_candidate_id(ids, nids, max_ids, vocab_size, tmp[i]);
}

static int32_t fill_observation_candidates(const TriadTokenizer *tok,
                                           int32_t vocab_size,
                                           uint64_t seed,
                                           int32_t *ids,
                                           int32_t max_ids) {
    int32_t nids = 0;
    if (tok) {
        add_candidate_id(ids, &nids, max_ids, vocab_size, tok->eos_id);
        const char *common[] = {
            " ", "\n", ".", ",", ":", ";", "!", "?",
            "a", "e", "o", "s", "n", "r", "t",
            " the", " and", " to", " of", " in", " is",
            " de", " que", " nao", " sim", " para", " com",
            " eu", " voce", " Triad", "Lang",
            NULL
        };
        for (int i = 0; common[i] && nids < max_ids; i++)
            add_encoded_candidates(tok, ids, &nids, max_ids, vocab_size, common[i]);
    }

    uint64_t state = seed ? seed : 0xC0FFEEULL;
    while (nids < max_ids) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        add_candidate_id(ids, &nids, max_ids, vocab_size,
                         (int32_t)(state % (uint64_t)vocab_size));
    }
    return nids;
}

static double token_readability_prior(const TriadTokenizer *tok, int32_t id) {
    if (!tok) return 0.0;
    const char *s = triad_tokenizer_id_to_str(tok, id);
    if (!s || !*s) return -2.0;
    if (id == tok->eos_id) return -0.5;
    if (strncmp(s, "<|", 2) == 0) return -3.0;

    int ascii = 0, alpha = 0, digit = 0, punct = 0, high = 0;
    for (const unsigned char *p = (const unsigned char*)s; *p; p++) {
        if (*p < 0x80) {
            ascii++;
            if ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z')) alpha++;
            else if (*p >= '0' && *p <= '9') digit++;
            else if (strchr(" \n.,:;!?'-_/()[]{}", *p)) punct++;
        } else {
            high++;
        }
    }

    int len = ascii + high;
    double prior = 0.0;
    if (ascii > 0 && high == 0) prior += 0.35;
    if (alpha > 0 || digit > 0) prior += 0.20;
    if (punct > 0 && len <= 4) prior += 0.15;
    if (s[0] == ' ' || strcmp(s, "\n") == 0) prior += 0.20;
    if (len > 16) prior -= 0.25;
    if (high > 0 && ascii == 0) prior -= 0.80;
    if (high > ascii) prior -= 0.45;
    return prior;
}

static void normalize_psi(TriadCplx *psi, int32_t N, double L) {
    double dx = L / N;
    double norm = 0.0;
    for (int32_t i = 0; i < N; i++)
        norm += psi[i].re * psi[i].re + psi[i].im * psi[i].im;
    norm = sqrt(norm * dx);
    if (norm < 1e-30) norm = 1.0;
    for (int32_t i = 0; i < N; i++) {
        psi[i].re /= norm;
        psi[i].im /= norm;
    }
}

static const GGUFTensorInfo *find_layer_tensor(GGUFFile *f, int32_t layer,
                                               const char *suffix) {
    char name[128];
    snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    return gguf_find_tensor(f, name);
}

static void init_layers(TriadGGUFSubstrate *rt) {
    rt->layers = calloc((size_t)rt->n_layers, sizeof(TriadGGUFLayer));
    if (!rt->layers) return;
    for (int32_t l = 0; l < rt->n_layers; l++) {
        TriadGGUFLayer *ly = &rt->layers[l];
        ly->attn_norm = find_layer_tensor(rt->gguf, l, "attn_norm.weight");
        ly->attn_qkv = find_layer_tensor(rt->gguf, l, "attn_qkv.weight");
        ly->attn_gate = find_layer_tensor(rt->gguf, l, "attn_gate.weight");
        ly->ssm_alpha = find_layer_tensor(rt->gguf, l, "ssm_alpha.weight");
        ly->ssm_beta = find_layer_tensor(rt->gguf, l, "ssm_beta.weight");
        ly->ssm_conv = find_layer_tensor(rt->gguf, l, "ssm_conv1d.weight");
        ly->ssm_out = find_layer_tensor(rt->gguf, l, "ssm_out.weight");
        ly->ssm_norm = find_layer_tensor(rt->gguf, l, "ssm_norm.weight");
        ly->ffn_norm = find_layer_tensor(rt->gguf, l, "post_attention_norm.weight");
        ly->ffn_gate = find_layer_tensor(rt->gguf, l, "ffn_gate.weight");
        ly->ffn_up = find_layer_tensor(rt->gguf, l, "ffn_up.weight");
        ly->ffn_down = find_layer_tensor(rt->gguf, l, "ffn_down.weight");
    }
}

static double mix_weight_row(TriadGGUFSubstrate *rt, const GGUFTensorInfo *t,
                             int64_t row, TriadCplx *psi, double strength) {
    if (!t || t->ndim != 2 || t->shape[0] != rt->dim || t->shape[1] <= 0)
        return 0.0;
    row %= t->shape[1];
    if (row < 0) row += t->shape[1];
    if (gguf_dequantize_row(rt->gguf, t, row, rt->layer_buf, rt->dim) != 0)
        return 0.0;

    double drive = tanh(dot_vec(rt->latent_buf, rt->layer_buf, rt->dim));
    vector_to_psi(rt->layer_buf, rt->dim, rt->field_buf, GRID_N, GRID_L);
    for (int32_t i = 0; i < GRID_N; i++) {
        psi[i].re += strength * drive * rt->field_buf[i].re;
        psi[i].im += strength * rt->field_buf[i].re * sin((double)i * 0.0245436926);
    }
    return drive;
}

static void apply_gguf_layers(TriadGGUFSubstrate *rt, TriadCplx *psi,
                              const TriadSolverC *base_cfg,
                              double *y_state,
                              int32_t pos, int32_t max_layers) {
    if (!rt || !rt->layers || max_layers <= 0) return;
    if (max_layers > rt->n_layers) max_layers = rt->n_layers;

    for (int32_t l = 0; l < max_layers; l++) {
        TriadGGUFLayer *ly = &rt->layers[l];
        psi_to_latent(psi, GRID_N, rt->latent_buf, rt->dim);

        double d0 = mix_weight_row(rt, ly->attn_qkv,
                                   (int64_t)pos * 131 + l * 17,
                                   psi, 0.030);
        double d1 = mix_weight_row(rt, ly->attn_gate,
                                   (int64_t)pos * 97 + l * 31,
                                   psi, 0.020);
        double d2 = mix_weight_row(rt, ly->ssm_alpha,
                                   (int64_t)pos + l,
                                   psi, 0.018);
        double d3 = mix_weight_row(rt, ly->ssm_beta,
                                   (int64_t)pos * 7 + l,
                                   psi, 0.018);
        double d4 = mix_weight_row(rt, ly->ffn_gate,
                                   (int64_t)pos * 53 + l * 11,
                                   psi, 0.016);
        double d5 = mix_weight_row(rt, ly->ffn_up,
                                   (int64_t)pos * 43 + l * 13,
                                   psi, 0.016);

        normalize_psi(psi, GRID_N, GRID_L);

        TriadSolverC cfg = *base_cfg;
        double drive = (d0 + d1 + d2 + d3 + d4 + d5) / 6.0;
        cfg.T = cfg.dt * 2.0;
        cfg.seed = base_cfg->seed + (uint64_t)(pos + 1) * 1009ULL + (uint64_t)l * 9176ULL;
        cfg.Lambda = base_cfg->Lambda * (1.0 + 0.08 * drive);
        cfg.alpha = base_cfg->alpha * (1.0 + 0.05 * fabs(d2 + d3));
        cfg.Gamma = base_cfg->Gamma * (1.0 + 0.03 * fabs(d1));
        cfg.f_FDT = base_cfg->f_FDT * (1.0 + 0.04 * fabs(d0));

        TriadSolverResult res = triad_solve_from_state(&cfg, psi, y_state);
        for (int32_t i = 0; i < GRID_N; i++) psi[i] = res.psi_final[i];
        if (y_state && res.y_final)
            memcpy(y_state, res.y_final, (size_t)cfg.M * cfg.N * sizeof(double));
        triad_solver_result_free(&res);
    }
}

static int ensure_safe_layer_cap(TriadSafeSubstrate *rt, int64_t need) {
    if (need <= rt->layer_cap) return 0;
    double *next = realloc(rt->layer_buf, (size_t)need * sizeof(double));
    if (!next) return -1;
    rt->layer_buf = next;
    rt->layer_cap = need;
    return 0;
}

static double mix_safe_weight_row(TriadSafeSubstrate *rt, const char *name,
                                  int64_t row, TriadCplx *psi, double strength) {
    TriadSafeTensorFile *file = NULL;
    const TriadSafeTensorInfo *t = triad_model_find_tensor(&rt->index, name, &file);
    if (!t || !file || t->ndim != 2 || t->shape[1] <= 0) return 0.0;
    if (ensure_safe_layer_cap(rt, t->shape[1]) != 0) return 0.0;
    row %= t->shape[0];
    if (row < 0) row += t->shape[0];
    if (triad_safetensors_read_row_f64(file, t, row, rt->layer_buf, t->shape[1]) < 0)
        return 0.0;

    int32_t dot_n = (t->shape[1] < rt->dim) ? (int32_t)t->shape[1] : rt->dim;
    double drive = tanh(dot_vec(rt->latent_buf, rt->layer_buf, dot_n));
    vector_to_psi(rt->layer_buf, (int32_t)t->shape[1], rt->field_buf, GRID_N, GRID_L);
    for (int32_t i = 0; i < GRID_N; i++) {
        psi[i].re += strength * drive * rt->field_buf[i].re;
        psi[i].im += strength * rt->field_buf[i].re * sin((double)i * 0.0245436926);
    }
    return drive;
}

static void safe_tensor_name(char *buf, size_t n, int32_t layer, const char *suffix) {
    snprintf(buf, n, "model.language_model.layers.%d.%s", layer, suffix);
}

static void mix_safe_layer_tensor(TriadSafeSubstrate *rt, double *drives, int *ndrives,
                                  int max_drives, char *name, size_t name_cap,
                                  int32_t layer, const char *suffix,
                                  int64_t row, TriadCplx *psi, double strength) {
    if (*ndrives >= max_drives) return;
    safe_tensor_name(name, name_cap, layer, suffix);
    TriadSafeTensorFile *file = NULL;
    const TriadSafeTensorInfo *t = triad_model_find_tensor(&rt->index, name, &file);
    if (!t || t->ndim != 2) return;
    drives[(*ndrives)++] = mix_safe_weight_row(rt, name, row, psi, strength);
}

static void apply_safe_layers(TriadSafeSubstrate *rt, TriadCplx *psi,
                              const TriadSolverC *base_cfg,
                              double *y_state,
                              int32_t pos, int32_t max_layers) {
    if (!rt || max_layers <= 0) return;
    if (max_layers > rt->n_layers) max_layers = rt->n_layers;

    char name[192];
    for (int32_t l = 0; l < max_layers; l++) {
        psi_to_latent(psi, GRID_N, rt->latent_buf, rt->dim);

        double d[16] = {0};
        int nd = 0;

        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "triad_attn.in_proj_qkv.weight",
                              (int64_t)pos * 131 + l * 17, psi, 0.030);
        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "triad_attn.in_proj_z.weight",
                              (int64_t)pos * 97 + l * 31, psi, 0.022);
        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "triad_attn.out_proj.weight",
                              (int64_t)pos * 89 + l * 23, psi, 0.022);
        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "triad_attn.in_proj_a.weight",
                              (int64_t)pos * 73 + l * 37, psi, 0.014);
        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "triad_attn.in_proj_b.weight",
                              (int64_t)pos * 71 + l * 41, psi, 0.014);
        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "triad_attn.conv1d.weight",
                              (int64_t)pos * 67 + l * 43, psi, 0.010);

        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "self_attn.q_proj.weight",
                              (int64_t)pos * 83 + l * 19, psi, 0.026);
        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "self_attn.k_proj.weight",
                              (int64_t)pos * 81 + l * 23, psi, 0.020);
        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "self_attn.v_proj.weight",
                              (int64_t)pos * 77 + l * 29, psi, 0.020);
        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "self_attn.o_proj.weight",
                              (int64_t)pos * 79 + l * 29, psi, 0.022);

        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "mlp.gate_proj.weight",
                              (int64_t)pos * 53 + l * 11, psi, 0.018);
        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "mlp.up_proj.weight",
                              (int64_t)pos * 43 + l * 13, psi, 0.018);
        mix_safe_layer_tensor(rt, d, &nd, 16, name, sizeof(name), l,
                              "mlp.down_proj.weight",
                              (int64_t)pos * 37 + l * 7, psi, 0.012);

        normalize_psi(psi, GRID_N, GRID_L);

        double drive = 0.0;
        for (int i = 0; i < nd; i++) drive += d[i];
        if (nd > 0) drive /= (double)nd;

        TriadSolverC cfg = *base_cfg;
        cfg.T = cfg.dt * 2.0;
        cfg.seed = base_cfg->seed + (uint64_t)(pos + 1) * 1009ULL + (uint64_t)l * 9176ULL;
        cfg.Lambda = base_cfg->Lambda * (1.0 + 0.08 * drive);
        cfg.alpha = base_cfg->alpha * (1.0 + 0.05 * fabs(drive));
        cfg.Gamma = base_cfg->Gamma * (1.0 + 0.03 * fabs(drive));
        cfg.f_FDT = base_cfg->f_FDT * (1.0 + 0.04 * fabs(drive));

        TriadSolverResult res = triad_solve_from_state(&cfg, psi, y_state);
        for (int32_t i = 0; i < GRID_N; i++) psi[i] = res.psi_final[i];
        if (y_state && res.y_final)
            memcpy(y_state, res.y_final, (size_t)cfg.M * cfg.N * sizeof(double));
        triad_solver_result_free(&res);
    }
}

static TriadCplx *decode_fft_buf = NULL;
static double    *decode_pwr     = NULL;
static double    *decode_kfreq   = NULL;

static void decode_init(int32_t N, double L) {
    decode_fft_buf = malloc(N * sizeof(TriadCplx));
    decode_pwr     = malloc(N * sizeof(double));
    decode_kfreq   = malloc(N * sizeof(double));
    double dx = L / N;
    triad_fftfreq(N, dx, decode_kfreq);
}

static void decode_cleanup(void) {
    free(decode_fft_buf);
    free(decode_pwr);
    free(decode_kfreq);
}

static void decode_to_logits(const TriadCplx *psi, int32_t N, double L,
                             double *logits, int32_t vocab_size) {
    for (int32_t i = 0; i < N; i++)
        decode_fft_buf[i] = psi[i];
    triad_fft_fn(N, decode_fft_buf, decode_fft_buf);

    for (int32_t i = 0; i < N; i++)
        decode_pwr[i] = decode_fft_buf[i].re * decode_fft_buf[i].re
                      + decode_fft_buf[i].im * decode_fft_buf[i].im;

    double k_min = decode_kfreq[0];
    double k_max = decode_kfreq[N - 1];
    double k_span = k_max - k_min;

    double max_l = -1e30;
    for (int32_t i = 0; i < vocab_size; i++) {
        double k_target = (double)i * (TRIAD_PI * 2.0 / L);

        double frac = (k_target - k_min) / k_span;
        int32_t idx = (int32_t)(frac * N);
        if (idx < 0) idx = 0;
        if (idx >= N) idx = N - 1;
        double pk = decode_pwr[idx];
        logits[i] = log(pk + 1e-30);
        if (logits[i] > max_l) max_l = logits[i];
    }
    for (int32_t i = 0; i < vocab_size; i++)
        logits[i] -= max_l;
}

static int32_t observe_next_token(TriadGGUFSubstrate *rt, const TriadTokenizer *tok,
                                  const TriadCplx *psi,
                                  int32_t N, int32_t candidate_count,
                                  const TriadSamplerConfig *sampler) {
    if (!rt || candidate_count <= 0) return -1;
    if (candidate_count > rt->vocab_size) candidate_count = rt->vocab_size;

    psi_to_latent(psi, N, rt->latent_buf, rt->dim);

    double *logits = malloc((size_t)candidate_count * sizeof(double));
    int32_t *ids = malloc((size_t)candidate_count * sizeof(int32_t));
    if (!logits || !ids) {
        free(logits);
        free(ids);
        return -1;
    }

    int32_t nids = fill_observation_candidates(tok, rt->vocab_size,
                                               sampler->seed, ids,
                                               candidate_count);
    for (int32_t c = 0; c < nids; c++) {
        int32_t id = ids[c];
        if (gguf_dequantize_row(rt->gguf, rt->out_w, id, rt->out_buf, rt->dim) != 0) {
            logits[c] = -1e300;
        } else {
            logits[c] = dot_vec(rt->latent_buf, rt->out_buf, rt->dim)
                      + token_readability_prior(tok, id);
        }
    }

    int32_t picked = triad_sample(logits, nids, sampler);
    int32_t result = (picked >= 0 && picked < nids) ? ids[picked] : -1;
    free(logits);
    free(ids);
    return result;
}

static int32_t observe_next_token_triad(TriadGGUFSubstrate *rt, const TriadCplx *psi,
                                       int32_t N, const TriadSamplerConfig *sampler) {
    if (!rt) return -1;
    psi_to_latent(psi, N, rt->latent_buf, rt->dim);

    double *logits = malloc((size_t)rt->vocab_size * sizeof(double));
    if (!logits) return -1;

    for (int32_t id = 0; id < rt->vocab_size; id++) {
        if (gguf_dequantize_row(rt->gguf, rt->out_w, id, rt->out_buf, rt->dim) != 0) {
            logits[id] = -1e300;
        } else {
            logits[id] = dot_vec(rt->latent_buf, rt->out_buf, rt->dim);
        }
    }

    int32_t result = triad_sample(logits, rt->vocab_size, sampler);
    free(logits);
    return result;
}

static int32_t observe_next_token_safe(TriadSafeSubstrate *rt, const TriadTokenizer *tok,
                                       const TriadCplx *psi,
                                       int32_t N, int32_t candidate_count,
                                       const TriadSamplerConfig *sampler) {
    if (!rt || candidate_count <= 0) return -1;
    if (candidate_count > rt->vocab_size) candidate_count = rt->vocab_size;

    psi_to_latent(psi, N, rt->latent_buf, rt->dim);
    double *logits = malloc((size_t)candidate_count * sizeof(double));
    int32_t *ids = malloc((size_t)candidate_count * sizeof(int32_t));
    if (!logits || !ids) {
        free(logits);
        free(ids);
        return -1;
    }

    int32_t nids = fill_observation_candidates(tok, rt->vocab_size,
                                               sampler->seed, ids,
                                               candidate_count);
    for (int32_t c = 0; c < nids; c++) {
        int32_t id = ids[c];
        if (triad_model_read_row_f64(&rt->index, "lm_head.weight", id,
                                     rt->out_buf, rt->dim) < 0) {
            logits[c] = -1e300;
        } else {
            logits[c] = dot_vec(rt->latent_buf, rt->out_buf, rt->dim)
                      + token_readability_prior(tok, id);
        }
    }

    int32_t picked = triad_sample(logits, nids, sampler);
    int32_t result = (picked >= 0 && picked < nids) ? ids[picked] : -1;
    free(logits);
    free(ids);
    return result;
}

static int32_t observe_next_token_safe_triad(TriadSafeSubstrate *rt, const TriadCplx *psi,
                                            int32_t N, const TriadSamplerConfig *sampler) {
    if (!rt) return -1;
    psi_to_latent(psi, N, rt->latent_buf, rt->dim);

    double *logits = malloc((size_t)rt->vocab_size * sizeof(double));
    if (!logits) return -1;
    for (int32_t id = 0; id < rt->vocab_size; id++) {
        if (triad_model_read_row_f64(&rt->index, "lm_head.weight", id,
                                     rt->out_buf, rt->dim) < 0) {
            logits[id] = -1e300;
        } else {
            logits[id] = dot_vec(rt->latent_buf, rt->out_buf, rt->dim);
        }
    }
    int32_t result = triad_sample(logits, rt->vocab_size, sampler);
    free(logits);
    return result;
}

static int backend_encode_gguf(TriadChatBackend *backend, int32_t token_id,
                               TriadCplx *psi, int32_t N, double L) {
    return encode_token_from_weights((TriadGGUFSubstrate*)backend->impl,
                                     token_id, psi, N, L);
}

static int32_t backend_observe_gguf(TriadChatBackend *backend, const TriadCplx *psi,
                                    int32_t N, int32_t candidates,
                                    const TriadSamplerConfig *sampler) {
    TriadGGUFSubstrate *rt = (TriadGGUFSubstrate*)backend->impl;
    if (candidates <= 0 || candidates >= rt->vocab_size)
        return observe_next_token_triad(rt, psi, N, sampler);
    return observe_next_token(rt, backend->tokenizer, psi, N, candidates, sampler);
}

static void backend_apply_gguf(TriadChatBackend *backend, TriadCplx *psi,
                               const TriadSolverC *cfg, double *y_state,
                               int32_t pos, int32_t max_layers) {
    apply_gguf_layers((TriadGGUFSubstrate*)backend->impl, psi, cfg, y_state,
                      pos, max_layers);
}

static int backend_encode_safe(TriadChatBackend *backend, int32_t token_id,
                               TriadCplx *psi, int32_t N, double L) {
    return encode_token_from_safetensors((TriadSafeSubstrate*)backend->impl,
                                         token_id, psi, N, L);
}

static int32_t backend_observe_safe(TriadChatBackend *backend, const TriadCplx *psi,
                                    int32_t N, int32_t candidates,
                                    const TriadSamplerConfig *sampler) {
    TriadSafeSubstrate *rt = (TriadSafeSubstrate*)backend->impl;
    if (candidates <= 0 || candidates >= rt->vocab_size)
        return observe_next_token_safe_triad(rt, psi, N, sampler);
    return observe_next_token_safe(rt, backend->tokenizer, psi, N, candidates, sampler);
}

static void backend_apply_safe(TriadChatBackend *backend, TriadCplx *psi,
                               const TriadSolverC *cfg, double *y_state,
                               int32_t pos, int32_t max_layers) {
    apply_safe_layers((TriadSafeSubstrate*)backend->impl, psi, cfg, y_state,
                      pos, max_layers);
}

static int utf8_next_cp(const unsigned char **p) {
    const unsigned char *s = *p;
    if (s[0] < 0x80) { *p = s + 1; return s[0]; }
    if ((s[0] & 0xE0) == 0xC0 && s[1]) {
        *p = s + 2;
        return ((s[0] & 0x1F) << 6) | (s[1] & 0x3F);
    }
    if ((s[0] & 0xF0) == 0xE0 && s[1] && s[2]) {
        *p = s + 3;
        return ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
    }
    if ((s[0] & 0xF8) == 0xF0 && s[1] && s[2] && s[3]) {
        *p = s + 4;
        return ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12)
             | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    }
    *p = s + 1;
    return s[0];
}

static void init_gpt2_byte_map(int *map, int nmap) {
    for (int i = 0; i < nmap; i++) map[i] = -1;
    int used[256] = {0};
    for (int b = '!'; b <= '~'; b++) { map[b] = b; used[b] = 1; }
    for (int b = 0xA1; b <= 0xAC; b++) { map[b] = b; used[b] = 1; }
    for (int b = 0xAE; b <= 0xFF; b++) { map[b] = b; used[b] = 1; }
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!used[b]) {
            int cp = 256 + n++;
            if (cp < nmap) map[cp] = b;
        }
    }
}

static void print_token_piece(const char *s) {
    static int byte_map[512];
    static int ready = 0;
    if (!ready) {
        init_gpt2_byte_map(byte_map, 512);
        ready = 1;
    }

    const unsigned char *p = (const unsigned char*)s;
    while (*p) {
        const unsigned char *before = p;
        int cp = utf8_next_cp(&p);
        if (cp >= 0 && cp < 512 && byte_map[cp] >= 0) {
            putchar(byte_map[cp]);
        } else {
            while (before < p) putchar(*before++);
        }
    }
}

static int64_t get_arch_int(GGUFFile *f, const char *arch, const char *suffix,
                             int64_t def) {
    char key[160];
    snprintf(key, sizeof(key), "%s.%s", arch, suffix);
    int64_t v = gguf_get_int(f, key, INT64_MIN);
    if (v != INT64_MIN) return v;

    if (strncmp(arch, "qwen", 4) == 0) {
        const char *aliases[] = { "qwen35", "qwen3", "qwen2", "qwen", NULL };
        for (int i = 0; aliases[i]; i++) {
            snprintf(key, sizeof(key), "%s.%s", aliases[i], suffix);
            v = gguf_get_int(f, key, INT64_MIN);
            if (v != INT64_MIN) return v;
        }
    }
    return def;
}

static TriadSolverC build_solver_config(GGUFFile *f) {
    TriadSolverC p;
    memset(&p, 0, sizeof(p));
    p.N     = GRID_N;
    p.L     = GRID_L;
    p.dt    = 0.01;
    p.T     = 2.0;
    p.hbar  = 1.0;
    p.m     = 1.0;
    p.omega = 0.05;
    p.seed  = (uint64_t)time(NULL);

    const char *arch = gguf_get_str(f, "general.architecture", "llama");

    int32_t n_layers = (int32_t)get_arch_int(f, arch, "block_count", 32);
    int32_t dim = (int32_t)get_arch_int(f, arch, "embedding_length", 4096);

    double cplx = (double)n_layers * dim / 4096.0 / 32.0;
    if (cplx < 0.5) cplx = 0.5;
    p.Lambda = -0.5 * cplx;
    p.alpha  = 0.15 * cplx;
    p.sigma  = 1.5;
    p.Gamma  = 0.05 * (1.0 + 0.5 * cplx);
    p.f_FDT  = 0.002 * cplx;

    p.M = 3;
    static double nu_buf[3]  = { 2.0, 0.5, 0.1 };
    static double lam_buf[3] = { -0.3, -0.2, -0.1 };
    p.nu  = nu_buf;
    p.lam = lam_buf;
    p.mode = 2;
    p.V_ext = "none";

    return p;
}

static TriadSolverC build_safe_solver_config(int32_t dim, int32_t n_layers) {
    TriadSolverC p;
    memset(&p, 0, sizeof(p));
    p.N     = GRID_N;
    p.L     = GRID_L;
    p.dt    = 0.01;
    p.T     = 2.0;
    p.hbar  = 1.0;
    p.m     = 1.0;
    p.omega = 0.05;
    p.seed  = (uint64_t)time(NULL);

    double cplx = (double)n_layers * dim / 4096.0 / 32.0;
    if (cplx < 0.5) cplx = 0.5;
    p.Lambda = -0.5 * cplx;
    p.alpha  = 0.15 * cplx;
    p.sigma  = 1.5;
    p.Gamma  = 0.05 * (1.0 + 0.5 * cplx);
    p.f_FDT  = 0.002 * cplx;

    p.M = 3;
    static double nu_buf[3]  = { 2.0, 0.5, 0.1 };
    static double lam_buf[3] = { -0.3, -0.2, -0.1 };
    p.nu  = nu_buf;
    p.lam = lam_buf;
    p.mode = 2;
    p.V_ext = "none";
    return p;
}

static int run_backend_chat_loop(TriadChatBackend *backend,
                                 TriadSolverC *scfg,
                                 double temperature,
                                 int32_t top_k,
                                 int32_t observe_candidates,
                                 int32_t active_layers,
                                 int32_t max_gen) {
    TriadCplx *psi = calloc(GRID_N, sizeof(TriadCplx));
    double *y_state = calloc((size_t)scfg->M * GRID_N, sizeof(double));
    double *logits = malloc((size_t)backend->vocab_size * sizeof(double));
    if (!psi || !y_state || !logits) {
        fprintf(stderr, "erro: sem memoria para buffers do chat\n");
        free(psi);
        free(y_state);
        free(logits);
        return 1;
    }
    decode_init(GRID_N, GRID_L);

    TriadSamplerConfig sampler = {
        .temperature = temperature,
        .top_k = top_k,
        .seed = (uint64_t)time(NULL)
    };

    printf("╔══════════════════════════════════════════╗\n");
    printf("║  TriadLang Chat  (%-18s) ║\n", backend->name);
    printf("║  Λ=%.2f α=%.2f Γ=%.2f grid=%d          ║\n",
           scfg->Lambda, scfg->alpha, scfg->Gamma, GRID_N);
    printf("╚══════════════════════════════════════════╝\n\n");

    char input[4096];
    int32_t pos = 0;
    while (1) {
        printf("> "); fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        size_t ilen = strlen(input);
        if (ilen > 0 && input[ilen - 1] == '\n') input[--ilen] = '\0';
        if (strcmp(input, "/quit") == 0 || strcmp(input, "/exit") == 0) break;
        if (ilen == 0) continue;

        int32_t ids[2048];
        int32_t n_tokens = triad_tokenizer_encode(backend->tokenizer, input, ids, 2048);
        if (n_tokens == 0) { printf("(vazio)\n"); continue; }

        fprintf(stderr, "\r[%d tokens]", n_tokens);
        for (int32_t i = 0; i < n_tokens; i++) {
            if (backend->encode_token(backend, ids[i], psi, GRID_N, GRID_L) != 0)
                encode_token_spectral(ids[i], psi, GRID_N, GRID_L);
            scfg->seed += pos + i;
            TriadSolverResult res = triad_solve_from_state(scfg, psi, y_state);
            for (int32_t j = 0; j < GRID_N; j++) psi[j] = res.psi_final[j];
            if (res.y_final)
                memcpy(y_state, res.y_final, (size_t)scfg->M * scfg->N * sizeof(double));
            triad_solver_result_free(&res);
            backend->apply_layers(backend, psi, scfg, y_state, pos, active_layers);
            pos++;
        }
        fprintf(stderr, "\r                    \r");

        printf("\n");
        for (int32_t g = 0; g < max_gen; g++) {
            sampler.seed++;
            int32_t next = backend->observe_next(backend, psi, GRID_N,
                                                 observe_candidates, &sampler);
            if (next < 0) {
                decode_to_logits(psi, GRID_N, GRID_L, logits, backend->vocab_size);
                next = triad_sample(logits, backend->vocab_size, &sampler);
            }
            if (next == backend->tokenizer->eos_id) break;
            if (next < 0 || next >= backend->tokenizer->vocab_size) break;

            print_token_piece(triad_tokenizer_id_to_str(backend->tokenizer, next));
            fflush(stdout);

            if (backend->encode_token(backend, next, psi, GRID_N, GRID_L) != 0)
                encode_token_spectral(next, psi, GRID_N, GRID_L);
            scfg->seed += pos + g;
            TriadSolverResult res = triad_solve_from_state(scfg, psi, y_state);
            for (int32_t j = 0; j < GRID_N; j++) psi[j] = res.psi_final[j];
            if (res.y_final)
                memcpy(y_state, res.y_final, (size_t)scfg->M * scfg->N * sizeof(double));
            triad_solver_result_free(&res);
            backend->apply_layers(backend, psi, scfg, y_state, pos, active_layers);
            pos++;
        }
        printf("\n\n");
    }

    decode_cleanup();
    free(psi);
    free(y_state);
    free(logits);
    printf("bye.\n");
    return 0;
}

static int is_safetensors_input(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    if (S_ISDIR(st.st_mode)) return 1;
    const char *ext = strstr(path, ".safetensors");
    return ext && ext[12] == '\0';
}

static void join_path(char *out, size_t n, const char *dir, const char *file) {
    size_t len = strlen(dir);
    snprintf(out, n, "%s%s%s", dir, (len > 0 && dir[len - 1] == '/') ? "" : "/", file);
}

static int run_safetensors_chat(const char *model_path, double temperature,
                                int32_t top_k, int32_t candidate_count,
                                int triad_vocab, int explicit_triad_vocab,
                                int32_t active_layers, int32_t max_gen) {
    char model_dir[4096], vocab_path[4096], merges_path[4096], tokenizer_config_path[4096];
    struct stat st;
    if (stat(model_path, &st) != 0) return 1;
    if (S_ISDIR(st.st_mode)) {
        snprintf(model_dir, sizeof(model_dir), "%s", model_path);
        join_path(vocab_path, sizeof(vocab_path), model_path, "vocab.json");
        join_path(merges_path, sizeof(merges_path), model_path, "merges.txt");
        join_path(tokenizer_config_path, sizeof(tokenizer_config_path),
                  model_path, "tokenizer_config.json");
    } else {
        snprintf(model_dir, sizeof(model_dir), "%s", model_path);
        char *slash = strrchr(model_dir, '/');
        if (slash) *slash = '\0';
        else snprintf(model_dir, sizeof(model_dir), ".");
        join_path(vocab_path, sizeof(vocab_path), model_dir, "vocab.json");
        join_path(merges_path, sizeof(merges_path), model_dir, "merges.txt");
        join_path(tokenizer_config_path, sizeof(tokenizer_config_path),
                  model_dir, "tokenizer_config.json");
    }

    fprintf(stderr, "Lendo safetensors %s...\n", model_dir);
    TriadSafeSubstrate rt;
    memset(&rt, 0, sizeof(rt));
    if (triad_model_index_load(&rt.index, model_dir) != 0) {
        fprintf(stderr, "falha ao abrir model.safetensors.index.json\n");
        return 1;
    }
    TriadSafeTensorFile *emb_file = NULL, *out_file = NULL;
    const TriadSafeTensorInfo *tok_emb =
        triad_model_find_tensor(&rt.index, "model.language_model.embed_tokens.weight", &emb_file);
    const TriadSafeTensorInfo *out_w =
        triad_model_find_tensor(&rt.index, "lm_head.weight", &out_file);
    if (!tok_emb || !out_w || tok_emb->ndim != 2 || out_w->ndim != 2) {
        fprintf(stderr, "erro: safetensors sem embed_tokens/lm_head no índice\n");
        triad_model_index_free(&rt.index);
        return 1;
    }
    rt.vocab_size = (int32_t)tok_emb->shape[0];
    rt.dim = (int32_t)tok_emb->shape[1];
    rt.n_layers = 32;
    if (active_layers < 0 || active_layers > rt.n_layers) active_layers = rt.n_layers;
    if (active_layers < 0) active_layers = 0;
    if (out_w->shape[0] != rt.vocab_size || out_w->shape[1] != rt.dim) {
        fprintf(stderr, "erro: lm_head incompatível com embedding\n");
        triad_model_index_free(&rt.index);
        return 1;
    }
    if (triad_vocab && !explicit_triad_vocab) {
        triad_vocab = 0;
        candidate_count = 2048;
    }
    if (!triad_vocab) {
        if (candidate_count < 256) candidate_count = 256;
        if (candidate_count > rt.vocab_size) candidate_count = rt.vocab_size;
    } else {
        candidate_count = rt.vocab_size;
    }

    rt.emb_buf = malloc((size_t)rt.dim * sizeof(double));
    rt.out_buf = malloc((size_t)rt.dim * sizeof(double));
    rt.latent_buf = malloc((size_t)rt.dim * sizeof(double));
    rt.layer_cap = (int64_t)rt.dim * 4;
    rt.layer_buf = malloc((size_t)rt.layer_cap * sizeof(double));
    rt.field_buf = malloc((size_t)GRID_N * sizeof(TriadCplx));
    if (!rt.emb_buf || !rt.out_buf || !rt.latent_buf || !rt.layer_buf || !rt.field_buf) {
        fprintf(stderr, "erro: sem memória para buffers safetensors\n");
        free(rt.emb_buf); free(rt.out_buf); free(rt.latent_buf);
        free(rt.layer_buf); free(rt.field_buf);
        triad_model_index_free(&rt.index);
        return 1;
    }

    TriadTokenizer *tok = triad_tokenizer_new();
    if (triad_tokenizer_load_vocab(tok, vocab_path) != 0 ||
        triad_tokenizer_load_merges(tok, merges_path) != 0) {
        fprintf(stderr, "erro: falha carregando vocab/merges (%s, %s)\n", vocab_path, merges_path);
        triad_tokenizer_free(tok);
        free(rt.emb_buf); free(rt.out_buf); free(rt.latent_buf);
        free(rt.layer_buf); free(rt.field_buf);
        triad_model_index_free(&rt.index);
        return 1;
    }
    triad_tokenizer_load_added_tokens(tok, tokenizer_config_path);
    tok->eos_id = 248044;
    tok->bos_id = 248045;

    fprintf(stderr, "  arquitetura: qwen3_5 safetensors  layers: %d  dim: %d  vocab: %d\n",
            rt.n_layers, rt.dim, rt.vocab_size);
    fprintf(stderr, "  motor: Triad PDE (P1+P2+P3) em grid %d\n", GRID_N);
    fprintf(stderr, "  modo: safetensors multi-shard BF16; camadas: %d/%d; leitura: %s (%d tokens)\n",
            active_layers, rt.n_layers, triad_vocab ? "vocabulário completo" : "candidatos", candidate_count);

    TriadSolverC scfg = build_safe_solver_config(rt.dim, rt.n_layers);
    fprintf(stderr, "  Λ=%.2f α=%.2f σ=%.1f Γ=%.2f T=%.1f M=%d\n\n",
            scfg.Lambda, scfg.alpha, scfg.sigma, scfg.Gamma, scfg.T, scfg.M);

    TriadChatBackend backend = {
        .name = "safetensors backend",
        .dim = rt.dim,
        .vocab_size = rt.vocab_size,
        .n_layers = rt.n_layers,
        .tokenizer = tok,
        .impl = &rt,
        .encode_token = backend_encode_safe,
        .observe_next = backend_observe_safe,
        .apply_layers = backend_apply_safe,
        .free = NULL,
    };
    int observe_candidates = triad_vocab ? 0 : candidate_count;
    int rc = run_backend_chat_loop(&backend, &scfg, temperature, top_k,
                                   observe_candidates, active_layers, max_gen);
    triad_tokenizer_free(tok);
    free(rt.emb_buf);
    free(rt.out_buf);
    free(rt.latent_buf);
    free(rt.layer_buf);
    free(rt.field_buf);
    triad_model_index_free(&rt.index);
    return rc;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "uso: triad-chat <modelo.gguf|models/> [--temp T] [--top-k K] [--triad-vocab] [--candidates N] [--layers N] [--max-gen N]\n");
        return 1;
    }

    const char *model_path = argv[1];
    double temperature = 0.7;
    int32_t top_k = 40;
    int32_t candidate_count = DEFAULT_CANDIDATES;
    int triad_vocab = 1;
    int explicit_triad_vocab = 0;
    int32_t active_layers = -1;
    int32_t max_gen = 64;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temperature = atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--candidates") == 0 && i + 1 < argc) {
            candidate_count = atoi(argv[++i]);
            triad_vocab = (candidate_count <= 0);
        }
        else if (strcmp(argv[i], "--triad-vocab") == 0) {
            candidate_count = 0;
            triad_vocab = 1;
            explicit_triad_vocab = 1;
        }
        else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) active_layers = atoi(argv[++i]);
        else if ((strcmp(argv[i], "--max-gen") == 0 || strcmp(argv[i], "--max-new") == 0) &&
                 i + 1 < argc) max_gen = atoi(argv[++i]);
    }
    if (max_gen < 1) max_gen = 1;

    if (is_safetensors_input(model_path)) {
        return run_safetensors_chat(model_path, temperature, top_k, candidate_count,
                                    triad_vocab, explicit_triad_vocab, active_layers, max_gen);
    }

    fprintf(stderr, "Lendo %s...\n", model_path);
    GGUFFile *f = gguf_open(model_path);
    if (!f) { fprintf(stderr, "falha ao abrir\n"); return 1; }

    const char *arch = gguf_get_str(f, "general.architecture", "llama");
    char key[128];

    int32_t dim = (int32_t)get_arch_int(f, arch, "embedding_length", 4096);
    int32_t n_layers = (int32_t)get_arch_int(f, arch, "block_count", 32);
    int32_t n_heads = (int32_t)get_arch_int(f, arch, "attention.head_count", 32);
    int32_t n_kv_heads = (int32_t)get_arch_int(f, arch, "attention.head_count_kv", 8);
    int32_t vocab_size = (int32_t)gguf_get_array_len(f, "tokenizer.ggml.tokens", 32000);
    if (!triad_vocab) {
        if (candidate_count < 256) candidate_count = 256;
        if (candidate_count > vocab_size) candidate_count = vocab_size;
    } else {
        candidate_count = vocab_size;
    }
    if (active_layers < 0 || active_layers > n_layers) active_layers = n_layers;
    if (active_layers < 0) active_layers = 0;

    fprintf(stderr, "  arquitetura: %s  layers: %d  dim: %d  heads: %d/%d  vocab: %d\n",
            arch, n_layers, dim, n_heads, n_kv_heads, vocab_size);
    fprintf(stderr, "  motor: Triad PDE (P1+P2+P3) em grid %d\n", GRID_N);
    fprintf(stderr, "  modo: pesos GGUF como memória/substrato; camadas: %d/%d; leitura: %s (%d tokens)\n",
            active_layers, n_layers, triad_vocab ? "vocabulário completo" : "candidatos",
            candidate_count);

    TriadGGUFSubstrate wrt;
    memset(&wrt, 0, sizeof(wrt));
    wrt.gguf = f;
    wrt.tok_emb = gguf_find_tensor(f, "token_embd.weight");
    wrt.out_w = gguf_find_tensor(f, "output.weight");
    wrt.n_layers = n_layers;
    wrt.dim = dim;
    wrt.vocab_size = vocab_size;
    if (!wrt.tok_emb || !wrt.out_w) {
        fprintf(stderr, "erro: GGUF sem token_embd.weight/output.weight; nao ha memória de pesos\n");
        gguf_close(f);
        return 1;
    }
    if (wrt.tok_emb->ndim != 2 || wrt.out_w->ndim != 2 ||
        wrt.tok_emb->shape[0] != dim || wrt.out_w->shape[0] != dim) {
        fprintf(stderr, "erro: tensores GGUF incompatíveis com dim=%d\n", dim);
        gguf_close(f);
        return 1;
    }
    wrt.emb_buf = malloc((size_t)dim * sizeof(double));
    wrt.out_buf = malloc((size_t)dim * sizeof(double));
    wrt.latent_buf = malloc((size_t)dim * sizeof(double));
    wrt.layer_buf = malloc((size_t)dim * sizeof(double));
    wrt.field_buf = malloc((size_t)GRID_N * sizeof(TriadCplx));
    if (!wrt.emb_buf || !wrt.out_buf || !wrt.latent_buf ||
        !wrt.layer_buf || !wrt.field_buf) {
        fprintf(stderr, "erro: sem memoria para buffers de substrato\n");
        free(wrt.emb_buf); free(wrt.out_buf); free(wrt.latent_buf);
        free(wrt.layer_buf); free(wrt.field_buf);
        gguf_close(f);
        return 1;
    }
    init_layers(&wrt);

    TriadTokenizer *tok = triad_tokenizer_new();
    {
        const GGUFKeyValue *vocab_kv = gguf_find_kv(f, "tokenizer.ggml.tokens");
        if (vocab_kv && vocab_kv->type == GGUF_TYPE_ARRAY &&
            vocab_kv->val.arr.elem_type == GGUF_TYPE_STRING) {
            const uint8_t *p = (const uint8_t*)vocab_kv->val.arr.data;
            for (uint64_t i = 0; i < vocab_kv->val.arr.len; i++) {
                uint64_t slen;
                memcpy(&slen, p, 8);
                p += 8;
                char *s = malloc((size_t)slen + 1);
                if (!s) break;
                memcpy(s, p, (size_t)slen);
                s[slen] = '\0';
                p += slen;
                triad_tokenizer_add_token(tok, (int32_t)i, s);
                free(s);
            }
        }
        const GGUFKeyValue *merges_kv = gguf_find_kv(f, "tokenizer.ggml.merges");
        if (merges_kv && merges_kv->type == GGUF_TYPE_ARRAY &&
            merges_kv->val.arr.elem_type == GGUF_TYPE_STRING) {
            const uint8_t *p = (const uint8_t*)merges_kv->val.arr.data;
            for (uint64_t i = 0; i < merges_kv->val.arr.len; i++) {
                uint64_t slen;
                memcpy(&slen, p, 8);
                p += 8;
                char *m = malloc((size_t)slen + 1);
                if (!m) break;
                memcpy(m, p, (size_t)slen);
                m[slen] = '\0';
                p += slen;
                char *sp = strchr(m, ' ');
                if (sp) { *sp = '\0'; }
                if (sp) triad_tokenizer_add_merge(tok, m, sp + 1, (int32_t)i);
                free(m);
            }
        }
    }
    snprintf(key, sizeof(key), "tokenizer.ggml.bos_token_id");
    tok->bos_id = (int32_t)gguf_get_int(f, key, 1);
    snprintf(key, sizeof(key), "tokenizer.ggml.eos_token_id");
    tok->eos_id = (int32_t)gguf_get_int(f, key, 2);

    TriadSolverC scfg = build_solver_config(f);
    fprintf(stderr, "  Λ=%.2f α=%.2f σ=%.1f Γ=%.2f T=%.1f M=%d\n\n",
            scfg.Lambda, scfg.alpha, scfg.sigma, scfg.Gamma, scfg.T, scfg.M);

    TriadChatBackend backend = {
        .name = "motor = Triad PDE",
        .dim = dim,
        .vocab_size = vocab_size,
        .n_layers = n_layers,
        .tokenizer = tok,
        .impl = &wrt,
        .encode_token = backend_encode_gguf,
        .observe_next = backend_observe_gguf,
        .apply_layers = backend_apply_gguf,
        .free = NULL,
    };
    int observe_candidates = triad_vocab ? 0 : candidate_count;
    int rc = run_backend_chat_loop(&backend, &scfg, temperature, top_k,
                                   observe_candidates, active_layers, max_gen);
    free(wrt.emb_buf);
    free(wrt.out_buf);
    free(wrt.latent_buf);
    free(wrt.layer_buf);
    free(wrt.field_buf);
    free(wrt.layers);
    triad_tokenizer_free(tok);
    gguf_close(f);
    return rc;
}
