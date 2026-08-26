#ifndef TRIAD_CHAT_BACKEND_H
#define TRIAD_CHAT_BACKEND_H

#include "triad_llm.h"
#include "triad_rt.h"
#include <stdint.h>

typedef struct TriadChatBackend TriadChatBackend;

struct TriadChatBackend {
    const char *name;
    int32_t dim;
    int32_t vocab_size;
    int32_t n_layers;
    TriadTokenizer *tokenizer;
    void *impl;

    int (*encode_token)(TriadChatBackend *backend, int32_t token_id,
                        TriadCplx *psi, int32_t N, double L);
    int32_t (*observe_next)(TriadChatBackend *backend, const TriadCplx *psi,
                            int32_t N, int32_t candidates,
                            const TriadSamplerConfig *sampler);
    void (*apply_layers)(TriadChatBackend *backend, TriadCplx *psi,
                         const TriadSolverC *cfg, double *y_state,
                         int32_t pos, int32_t max_layers);
    void (*free)(TriadChatBackend *backend);
};

#endif
