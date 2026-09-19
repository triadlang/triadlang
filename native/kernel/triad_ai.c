#include "triad_kernel.h"
#include "triad_ai.h"
#include "triad_mm.h"
#include "triad_sched.h"
#include <string.h>

static AiObservation observations[AI_MAX_OBSERVATIONS];
static uint64_t obs_count = 0;
static AiSession sessions[AI_MAX_SESSIONS];

static uint64_t simple_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (uint8_t)*s;
        h *= 1099511628211ULL;
        s++;
    }
    return h;
}

void triad_ai_init(void) {
    for (int i = 0; i < AI_MAX_OBSERVATIONS; i++) {
        observations[i].source[0] = 0;
        observations[i].text[0] = 0;
    }
    obs_count = 0;
    for (int i = 0; i < AI_MAX_SESSIONS; i++) {
        sessions[i].active = false;
    }
}

void triad_ai_observe(uint32_t proc_id, const char *source, const char *text) {
    uint64_t idx = obs_count % AI_MAX_OBSERVATIONS;
    AiObservation *o = &observations[idx];
    if (!source) source = "";
    if (!text) text = "";
    o->proc_id = proc_id;
    o->tick = triad_sched_get_ticks();
    int k = 0;
    while (source[k] && k < 127) { o->source[k] = source[k]; k++; }
    o->source[k] = 0;
    k = 0;
    while (text[k] && k < 511) { o->text[k] = text[k]; k++; }
    o->text[k] = 0;
    o->token_count = 0;
    while (text[o->token_count] && o->token_count < 512) o->token_count++;
    for (int i = 0; i < AI_EMBED_DIM; i++) {
        o->embedding[i] = (float)((simple_hash(o->text) >> (i % 64)) & 1);
    }
    obs_count++;
}

int triad_ai_embed(const char *text, float *out, uint32_t dim) {
    if (!text || !out) return -1;
    if (dim < AI_EMBED_DIM) return -1;
    uint64_t h = simple_hash(text);
    for (int i = 0; i < AI_EMBED_DIM; i++) {
        out[i] = (float)((h >> (i % 64)) & 1);
    }
    return AI_EMBED_DIM;
}

int triad_ai_recall(const char *query, AiObservation *out, int max_n) {
    float q_emb[AI_EMBED_DIM];
    if (!query || !out || max_n <= 0) return 0;
    if (triad_ai_embed(query, q_emb, AI_EMBED_DIM) < 0) return 0;
    int found = 0;
    for (int i = 0; i < AI_MAX_OBSERVATIONS && found < max_n; i++) {
        if (observations[i].source[0] == 0) continue;
        float dot = 0;
        for (int j = 0; j < AI_EMBED_DIM; j++) {
            dot += q_emb[j] * observations[i].embedding[j];
        }
        if (dot > 0) {
            out[found] = observations[i];
            found++;
        }
    }
    return found;
}

int triad_ai_query(const char *query, char *out, uint32_t out_len) {
    AiObservation results[8];
    if (!query || !out || out_len == 0) return -1;
    int n = triad_ai_recall(query, results, 8);
    if (n == 0) {
        const char fallback[] = "no observations match this query";
        int k = 0;
        while (fallback[k] && k < (int)out_len - 1) { out[k] = fallback[k]; k++; }
        out[k] = 0;
        return 0;
    }
    int pos = 0;
    for (int i = 0; i < n && pos < (int)out_len - 1; i++) {
        int k = 0;
        while (results[i].text[k] && pos < (int)out_len - 1) {
            out[pos] = results[i].text[k];
            pos++; k++;
        }
        if (pos < (int)out_len - 1) {
            out[pos] = '\n';
            pos++;
        }
    }
    out[pos] = 0;
    return n;
}

int triad_ai_tool_use(uint32_t proc_id, const char *tool, const char *args,
                      char *out, uint32_t out_len, bool require_confirm) {
    (void)proc_id;
    if (!tool || !out || out_len == 0) return -1;
    if (require_confirm) {
        const char msg[] = "confirmation required for destructive tool use";
        int k = 0;
        while (msg[k] && k < (int)out_len - 1) { out[k] = msg[k]; k++; }
        out[k] = 0;
        return -1;
    }
    if (strcmp(tool, "recall") == 0) {
        return triad_ai_query(args, out, out_len);
    }
    const char unknown[] = "unknown tool";
    int k = 0;
    while (unknown[k] && k < (int)out_len - 1) { out[k] = unknown[k]; k++; }
    out[k] = 0;
    return -1;
}

uint64_t triad_ai_observation_count(void) {
    return obs_count;
}