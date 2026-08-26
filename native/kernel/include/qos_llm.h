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
    double *weights;     // Campo quântico como pesos
    double *biases;
    int in_dim;
    int out_dim;
} QosLayer;

typedef struct {
    // Embeddings (representação do sistema)
    double *embed_tokens;   // [vocab, embed]
    double *embed_pos;      // [context, embed]
    
    // Camadas do transformer (simplificado)
    QosLayer *attn_q;        // Query
    QosLayer *attn_k;        // Key
    QosLayer *attn_v;        // Value
    QosLayer *attn_out;      // Output
    
    // Field Memory (P2-inspired)
    double *memory_bank;    // [memory_slots, hidden]
    double *memory_wells;    // Potenciais de memória
    
    // Contexto do sistema
    char *context_buffer;
    int context_len;
    int context_cap;
    
    // Integração com processos quânticos
    int bound_pid;           // Processo quântico associado
    double influence_kappa;  // Quanto o LLM influencia o campo
    
} QosLLM;

void qos_llm_init(void);
QosLLM *qos_llm_create(int embed_dim, int hidden_dim, int context_len);
void qos_llm_destroy(QosLLM *llm);

// Treinamento online (absorve tudo)
void qos_llm_absorb(QosLLM *llm, const char *text, int len);
void qos_llm_absorb_process(QosLLM *llm, QosProcess *proc);

// Geração
int qos_llm_generate(QosLLM *llm, const char *prompt, char *output, int max_len);
void qos_llm_generate_stream(QosLLM *llm, const char *prompt, 
                             void (*callback)(const char *token));

// Integração com campo quântico
void qos_llm_bind_process(QosLLM *llm, int pid, double kappa);
void qos_llm_apply_to_field(QosLLM *llm, QosProcess *proc);

// Auto-otimização (sem calibração externa!)
void qos_llm_self_organize(QosLLM *llm);

// Funções helper
double *qos_llm_encode(QosLLM *llm, const char *text, int len);
void qos_llm_decode(QosLLM *llm, double *embed, char *output, int max_len);

// Sugestões baseadas no estado do sistema
int qos_llm_suggest_command(QosLLM *llm, const char *context, 
                            char *suggestion, int max_len);

// Persistência
int qos_llm_save(QosLLM *llm, const char *path);
int qos_llm_load(QosLLM *llm, const char *path);

#endif