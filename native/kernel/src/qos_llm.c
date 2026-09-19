#include "qos_llm.h"
#include "qos_process.h"
#include "triad_rt.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static QosLLM *g_kernel_llm = NULL;

static double randn_llm(void) {
    static uint64_t state = 0xCAFEBABE98765432ULL;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    
    double u1 = (double)(state >> 11) / (double)(1ULL << 53);
    double u2 = (double)(state >> 33) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double softmax(double *x, int i, int n) {
    double max_val = x[0];
    for (int j = 1; j < n; j++) {
        if (x[j] > max_val) max_val = x[j];
    }
    
    double sum = 0;
    for (int j = 0; j < n; j++) {
        sum += exp(x[j] - max_val);
    }
    
    return exp(x[i] - max_val) / sum;
}

static void layer_forward(QosLayer *layer, double *in, double *out) {
    for (int i = 0; i < layer->out_dim; i++) {
        double sum = layer->biases[i];
        for (int j = 0; j < layer->in_dim; j++) {
            sum += layer->weights[i * layer->in_dim + j] * in[j];
        }
        out[i] = sum;
    }
}

static void relu(double *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

void qos_llm_init(void) {
    if (g_kernel_llm) return;
    
    g_kernel_llm = qos_llm_create(QOS_LLM_EMBED, QOS_LLM_HIDDEN, QOS_LLM_CONTEXT);
}

QosLLM *qos_llm_create(int embed_dim, int hidden_dim, int context_len) {
    QosLLM *llm = (QosLLM *)malloc(sizeof(QosLLM));
    if (!llm) return NULL;
    
    memset(llm, 0, sizeof(QosLLM));
    
    llm->embed_tokens = (double *)malloc(QOS_LLM_VOCAB * embed_dim * sizeof(double));
    llm->embed_pos = (double *)malloc(context_len * embed_dim * sizeof(double));
    
    for (int i = 0; i < QOS_LLM_VOCAB * embed_dim; i++) {
        llm->embed_tokens[i] = randn_llm() * 0.1;
    }
    for (int i = 0; i < context_len * embed_dim; i++) {
        llm->embed_pos[i] = randn_llm() * 0.1;
    }
    
    llm->attn_q = (QosLayer *)malloc(sizeof(QosLayer));
    llm->attn_k = (QosLayer *)malloc(sizeof(QosLayer));
    llm->attn_v = (QosLayer *)malloc(sizeof(QosLayer));
    llm->attn_out = (QosLayer *)malloc(sizeof(QosLayer));
    
    llm->attn_q->weights = (double *)malloc(hidden_dim * hidden_dim * sizeof(double));
    llm->attn_q->biases = (double *)malloc(hidden_dim * sizeof(double));
    llm->attn_q->in_dim = hidden_dim;
    llm->attn_q->out_dim = hidden_dim;
    
    llm->attn_k->weights = (double *)malloc(hidden_dim * hidden_dim * sizeof(double));
    llm->attn_k->biases = (double *)malloc(hidden_dim * sizeof(double));
    llm->attn_k->in_dim = hidden_dim;
    llm->attn_k->out_dim = hidden_dim;
    
    llm->attn_v->weights = (double *)malloc(hidden_dim * hidden_dim * sizeof(double));
    llm->attn_v->biases = (double *)malloc(hidden_dim * sizeof(double));
    llm->attn_v->in_dim = hidden_dim;
    llm->attn_v->out_dim = hidden_dim;
    
    llm->attn_out->weights = (double *)malloc(hidden_dim * hidden_dim * sizeof(double));
    llm->attn_out->biases = (double *)malloc(hidden_dim * sizeof(double));
    llm->attn_out->in_dim = hidden_dim;
    llm->attn_out->out_dim = hidden_dim;
    
    for (int i = 0; i < hidden_dim * hidden_dim; i++) {
        llm->attn_q->weights[i] = randn_llm() * 0.1;
        llm->attn_k->weights[i] = randn_llm() * 0.1;
        llm->attn_v->weights[i] = randn_llm() * 0.1;
        llm->attn_out->weights[i] = randn_llm() * 0.1;
    }
    for (int i = 0; i < hidden_dim; i++) {
        llm->attn_q->biases[i] = 0;
        llm->attn_k->biases[i] = 0;
        llm->attn_v->biases[i] = 0;
        llm->attn_out->biases[i] = 0;
    }
    
    llm->memory_bank = (double *)malloc(QOS_LLM_MEMORY * hidden_dim * sizeof(double));
    llm->memory_wells = (double *)malloc(QOS_LLM_MEMORY * sizeof(double));
    
    for (int i = 0; i < QOS_LLM_MEMORY * hidden_dim; i++) {
        llm->memory_bank[i] = randn_llm() * 0.1;
    }
    for (int i = 0; i < QOS_LLM_MEMORY; i++) {
        llm->memory_wells[i] = (double)i / QOS_LLM_MEMORY;
    }
    llm->context_cap = context_len * 16;
    llm->context_buffer = (char *)malloc(llm->context_cap);
    llm->context_len = 0;
    llm->context_buffer[0] = '\0';
    
    llm->bound_pid = -1;
    llm->influence_kappa = 0.1;
    
    return llm;
}

void qos_llm_destroy(QosLLM *llm) {
    if (!llm) return;
    
    if (llm->embed_tokens) free(llm->embed_tokens);
    if (llm->embed_pos) free(llm->embed_pos);
    if (llm->attn_q) {
        free(llm->attn_q->weights);
        free(llm->attn_q->biases);
        free(llm->attn_q);
    }
    if (llm->attn_k) {
        free(llm->attn_k->weights);
        free(llm->attn_k->biases);
        free(llm->attn_k);
    }
    if (llm->attn_v) {
        free(llm->attn_v->weights);
        free(llm->attn_v->biases);
        free(llm->attn_v);
    }
    if (llm->attn_out) {
        free(llm->attn_out->weights);
        free(llm->attn_out->biases);
        free(llm->attn_out);
    }
    if (llm->memory_bank) free(llm->memory_bank);
    if (llm->memory_wells) free(llm->memory_wells);
    if (llm->context_buffer) free(llm->context_buffer);
    
    free(llm);
}

void qos_llm_absorb(QosLLM *llm, const char *text, int len) {
    if (!llm || !text || len <= 0) return;
    
    int new_len = llm->context_len + len;
    if (new_len >= llm->context_cap) {
        int shift = new_len - llm->context_cap + 256;
        memmove(llm->context_buffer, llm->context_buffer + shift, llm->context_len - shift);
        llm->context_len -= shift;
    }
    
    memcpy(llm->context_buffer + llm->context_len, text, len);
    llm->context_len += len;
    llm->context_buffer[llm->context_len] = '\0';
    
    for (int i = 0; i < len && i < QOS_LLM_CONTEXT; i++) {
        int ch = (unsigned char)text[i];
        if (ch < 0 || ch >= QOS_LLM_VOCAB) ch = 0;
        double *emb = llm->embed_tokens + ch * QOS_LLM_EMBED;
        double *pos = llm->embed_pos + i * QOS_LLM_EMBED;
        for (int j = 0; j < QOS_LLM_EMBED; j++) {
            double delta = (pos[j] - emb[j]) * 0.01;
            emb[j] += delta + randn_llm() * 0.001;
        }
    }
    qos_llm_self_organize(llm);
}

void qos_llm_absorb_process(QosLLM *llm, QosProcess *proc) {
    if (!llm || !proc) return;
    
    double energy = proc->energy;
    double cryst = proc->crystallinity;
    double kstar = proc->k_star;
    
    char buf[256];
    snprintf(buf, sizeof(buf), "[proc:%d energy:%.2f cryst:%.2f k:%.2f] ",
             proc->pid, energy, cryst, kstar);
    qos_llm_absorb(llm, buf, strlen(buf));
    
    if (cryst > 0.9 && llm->bound_pid >= 0) {
        double *well = llm->memory_wells;
        for (int i = 0; i < QOS_LLM_MEMORY; i++) {
            well[i] = well[i] * 0.95 + cryst * 0.05;
        }
    }
}

void qos_llm_bind_process(QosLLM *llm, int pid, double kappa) {
    if (!llm) return;
    llm->bound_pid = pid;
    llm->influence_kappa = kappa;
}

void qos_llm_apply_to_field(QosLLM *llm, QosProcess *proc) {
    if (!llm || !proc || llm->bound_pid != proc->pid) return;
    
    double influence = llm->influence_kappa;
    for (int i = 0; i < proc->N && i < QOS_LLM_EMBED; i++) {
        double emb_sum = 0;
        for (int v = 0; v < QOS_LLM_VOCAB; v++) {
            emb_sum += llm->embed_tokens[v * QOS_LLM_EMBED + i];
        }
        emb_sum /= QOS_LLM_VOCAB;
        
        proc->psi[i].re += influence * emb_sum * 0.001;
    }
}

void qos_llm_self_organize(QosLLM *llm) {
    if (!llm) return;
    
    for (int v = 0; v < QOS_LLM_VOCAB; v++) {
        double *emb = llm->embed_tokens + v * QOS_LLM_EMBED;
        double norm = 0;
        for (int i = 0; i < QOS_LLM_EMBED; i++) {
            norm += emb[i] * emb[i];
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (int i = 0; i < QOS_LLM_EMBED; i++) {
                emb[i] /= norm;
            }
        }
    }
    
    for (int i = 0; i < QOS_LLM_MEMORY; i++) {
        double *mem = llm->memory_bank + i * QOS_LLM_HIDDEN;
        for (int j = 0; j < QOS_LLM_HIDDEN && j < QOS_LLM_EMBED; j++) {
            double sum = 0;
            for (int v = 0; v < QOS_LLM_VOCAB; v++) {
                sum += llm->embed_tokens[v * QOS_LLM_EMBED + j];
            }
            mem[j] = mem[j] * 0.99 + (sum / QOS_LLM_VOCAB) * 0.01;
        }
        
        llm->memory_wells[i] *= 0.999;
    }
}

double *qos_llm_encode(QosLLM *llm, const char *text, int len) {
    if (!llm || !text) return NULL;
    
    double *embed = (double *)malloc(QOS_LLM_HIDDEN * sizeof(double));
    if (!embed) return NULL;
    
    memset(embed, 0, QOS_LLM_HIDDEN * sizeof(double));
    
    for (int i = 0; i < len && i < QOS_LLM_CONTEXT; i++) {
        int ch = (unsigned char)text[i];
        if (ch < 0 || ch >= QOS_LLM_VOCAB) ch = 0;
        
        double *tok_emb = llm->embed_tokens + ch * QOS_LLM_EMBED;
        double *pos_emb = llm->embed_pos + i * QOS_LLM_EMBED;
        
        for (int j = 0; j < QOS_LLM_EMBED && j < QOS_LLM_HIDDEN; j++) {
            embed[j] += tok_emb[j] + pos_emb[j];
        }
    }
    
    double norm = 0;
    for (int i = 0; i < QOS_LLM_HIDDEN; i++) {
        norm += embed[i] * embed[i];
    }
    norm = sqrt(norm);
    if (norm > 1e-10) {
        for (int i = 0; i < QOS_LLM_HIDDEN; i++) {
            embed[i] /= norm;
        }
    }
    return embed;
}

void qos_llm_decode(QosLLM *llm, double *embed, char *output, int max_len) {
    if (!llm || !embed || !output || max_len <= 0) return;
    for (int pos = 0; pos < max_len - 1; pos++) {
        double best_score = -1e30;
        int best_token = 0;
        
        for (int v = 0; v < QOS_LLM_VOCAB; v++) {
            double *tok_emb = llm->embed_tokens + v * QOS_LLM_EMBED;
            double score = 0;
            
            for (int j = 0; j < QOS_LLM_EMBED && j < QOS_LLM_HIDDEN; j++) {
                score += embed[j] * tok_emb[j];
            }
            
            double *pos_emb = llm->embed_pos + pos * QOS_LLM_EMBED;
            for (int j = 0; j < QOS_LLM_EMBED && j < QOS_LLM_HIDDEN; j++) {
                score += pos_emb[j] * 0.1;
            }
            
            if (score > best_score) {
                best_score = score;
                best_token = v;
            }
        }
        
        if (randn_llm() > 0.5) {
            best_token = (best_token + (int)(randn_llm() * 10)) % QOS_LLM_VOCAB;
            if (best_token < 0) best_token += QOS_LLM_VOCAB;
        }
        
        output[pos] = (char)best_token;
        
        double *tok_emb = llm->embed_tokens + best_token * QOS_LLM_EMBED;
        for (int j = 0; j < QOS_LLM_EMBED && j < QOS_LLM_HIDDEN; j++) {
            embed[j] = embed[j] * 0.7 + tok_emb[j] * 0.3;
        }
        
        if (best_token == ' ' || best_token == '\n' || best_token == 0) {
            output[pos + 1] = '\0';
            break;
        }
    }
    
    output[max_len - 1] = '\0';
}

int qos_llm_generate(QosLLM *llm, const char *prompt, char *output, int max_len) {
    if (!llm || !prompt || !output || max_len <= 0) return 0;
    
    qos_llm_absorb(llm, prompt, strlen(prompt));
    double *embed = qos_llm_encode(llm, llm->context_buffer, llm->context_len);
    if (!embed) return 0;
    double *q = (double *)malloc(QOS_LLM_HIDDEN * sizeof(double));
    double *k = (double *)malloc(QOS_LLM_HIDDEN * sizeof(double));
    double *v = (double *)malloc(QOS_LLM_HIDDEN * sizeof(double));
    
    layer_forward(llm->attn_q, embed, q);
    layer_forward(llm->attn_k, embed, k);
    layer_forward(llm->attn_v, embed, v);
    
    double scale = 1.0 / sqrt((double)QOS_LLM_HIDDEN);
    double attn_sum = 0;
    double *attn_weights = (double *)malloc(QOS_LLM_MEMORY * sizeof(double));
    
    for (int i = 0; i < QOS_LLM_MEMORY; i++) {
        double score = 0;
        double *mem = llm->memory_bank + i * QOS_LLM_HIDDEN;
        for (int j = 0; j < QOS_LLM_HIDDEN; j++) {
            score += q[j] * mem[j];
        }
        attn_weights[i] = exp(score * scale);
        attn_sum += attn_weights[i];
    }
    
    double *attn_out = (double *)malloc(QOS_LLM_HIDDEN * sizeof(double));
    memset(attn_out, 0, QOS_LLM_HIDDEN * sizeof(double));
    
    for (int i = 0; i < QOS_LLM_MEMORY; i++) {
        double w = attn_weights[i] / attn_sum;
        double *mem = llm->memory_bank + i * QOS_LLM_HIDDEN;
        for (int j = 0; j < QOS_LLM_HIDDEN; j++) {
            attn_out[j] += w * mem[j];
        }
    }
    
    for (int j = 0; j < QOS_LLM_HIDDEN; j++) {
        embed[j] = embed[j] * 0.5 + attn_out[j] * 0.5 + v[j] * 0.3;
    }
    relu(embed, QOS_LLM_HIDDEN);
    qos_llm_decode(llm, embed, output, max_len);
    free(q);
    free(k);
    free(v);
    free(attn_weights);
    free(attn_out);
    free(embed);
    
    return 1;
}

int qos_llm_suggest_command(QosLLM *llm, const char *context,
                            char *suggestion, int max_len) {
    if (!llm || !context || !suggestion || max_len <= 0) return 0;
    
    double *embed = qos_llm_encode(llm, context, strlen(context));
    if (!embed) return 0;
    const char *commands[] = {
        "ls", "cat", "run", "eval", "mem", "help", "clear", "exit"
    };
    int n_commands = 8;
    
    double best_score = -1e30;
    int best_cmd = 0;
    
    for (int i = 0; i < n_commands; i++) {
        double *cmd_emb = qos_llm_encode(llm, commands[i], strlen(commands[i]));
        if (!cmd_emb) continue;
        
        double score = 0;
        for (int j = 0; j < QOS_LLM_HIDDEN; j++) {
            score += embed[j] * cmd_emb[j];
        }
        
        if (score > best_score) {
            best_score = score;
            best_cmd = i;
        }
        
        free(cmd_emb);
    }
    
    snprintf(suggestion, max_len, "%s", commands[best_cmd]);
    
    free(embed);
    return 1;
}

int qos_llm_save(QosLLM *llm, const char *path) {
    (void)llm;
    (void)path;
    return 0;
}

int qos_llm_load(QosLLM *llm, const char *path) {
    (void)llm;
    (void)path;
    return 0;
}