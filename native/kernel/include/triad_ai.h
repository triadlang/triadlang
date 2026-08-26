#ifndef TRIAD_AI_H
#define TRIAD_AI_H

#include "triad_kernel.h"

#define AI_MAX_OBSERVATIONS 4096
#define AI_EMBED_DIM        256
#define AI_MAX_SESSIONS     8

typedef struct {
    char     source[128];
    char     text[512];
    uint32_t proc_id;
    uint64_t tick;
    float    embedding[AI_EMBED_DIM];
    uint32_t token_count;
} AiObservation;

typedef struct {
    bool            active;
    uint32_t        proc_id;
    char            query[512];
    char            response[1024];
    bool            ready;
} AiSession;

void triad_ai_init(void);
void triad_ai_observe(uint32_t proc_id, const char *source, const char *text);
int  triad_ai_query(const char *query, char *out, uint32_t out_len);
int  triad_ai_tool_use(uint32_t proc_id, const char *tool, const char *args,
                       char *out, uint32_t out_len, bool require_confirm);
uint64_t triad_ai_observation_count(void);
int  triad_ai_embed(const char *text, float *out, uint32_t dim);
int  triad_ai_recall(const char *query, AiObservation *out, int max_n);

#endif