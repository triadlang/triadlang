#ifndef QOS_LLM_H
#define QOS_LLM_H

#include <stdint.h>
#include <stddef.h>
#include "qos_process.h"

#define QOS_LLM_VOCAB 256
#define QOS_LLM_EMBED 64
#define QOS_LLM_HIDDEN 128
#define QOS_LLM_CONTEXT 512
#define QOS_LLM_MEMORY 16

typedef struct {
    double *weights;
    double *biases;
    int in_dim;
    int out_dim;
} QosLayer;

typedef struct {
    double *embed_tokens;
    double *embed_pos;
    QosLayer *attn_q;
    QosLayer *attn_k;
    QosLayer *attn_v;
    QosLayer *attn_out;
    double *memory_bank;
    double *memory_wells;
    char *context_buffer;
    int context_len;
    int context_cap;
    int bound_pid;
    double influence_kappa;
} QosLLM;

void qos_llm_init(void);
QosLLM *qos_llm_create(int embed_dim, int hidden_dim, int context_len);
void qos_llm_destroy(QosLLM *llm);
void qos_llm_absorb(QosLLM *llm, const char *text, int len);
void qos_llm_absorb_process(QosLLM *llm, QosProcess *proc);
int qos_llm_generate(QosLLM *llm, const char *prompt, char *output, int max_len);
void qos_llm_generate_stream(QosLLM *llm, const char *prompt,
                             void (*callback)(const char *token));
void qos_llm_bind_process(QosLLM *llm, int pid, double kappa);
void qos_llm_apply_to_field(QosLLM *llm, QosProcess *proc);
void qos_llm_self_organize(QosLLM *llm);
double *qos_llm_encode(QosLLM *llm, const char *text, int len);
void qos_llm_decode(QosLLM *llm, double *embed, char *output, int max_len);
int qos_llm_suggest_command(QosLLM *llm, const char *context,
                            char *suggestion, int max_len);
int qos_llm_save(QosLLM *llm, const char *path);
int qos_llm_load(QosLLM *llm, const char *path);

#endif
