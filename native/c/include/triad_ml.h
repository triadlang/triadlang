#ifndef TRIAD_ML_H
#define TRIAD_ML_H

#include <stdint.h>
#include <stddef.h>

typedef struct TriadTensor TriadTensor;

typedef void (*TriadGradFn)(TriadTensor *out);

struct TriadTensor {
    int32_t      refcount;
    int32_t      ndim;
    int32_t     *shape;
    int64_t      size;
    double      *data;
    double      *grad;
    int          requires_grad;

    TriadGradFn  grad_fn;
    int32_t      nchildren;
    TriadTensor **children;
    void        *_ctx;
    void       (*ctx_free)(void*);
};

TriadTensor *triad_tensor_new(int32_t ndim, const int32_t *shape, int requires_grad);
TriadTensor *triad_tensor_from_data(int32_t ndim, const int32_t *shape,
                                     const double *data, int requires_grad);
TriadTensor *triad_tensor_scalar(double val, int requires_grad);
TriadTensor *triad_tensor_zeros(int32_t ndim, const int32_t *shape, int rg);
TriadTensor *triad_tensor_ones(int32_t ndim, const int32_t *shape, int rg);
TriadTensor *triad_tensor_randn(int32_t ndim, const int32_t *shape, int rg);
TriadTensor *triad_tensor_rand(int32_t ndim, const int32_t *shape, int rg);
TriadTensor *triad_tensor_eye(int32_t n);
TriadTensor *triad_tensor_arange(double start, double stop, double step);
TriadTensor *triad_tensor_linspace(double start, double stop, int32_t steps);

void triad_tensor_free(TriadTensor *t);
void triad_tensor_retain(TriadTensor *t);

void triad_tensor_backward(TriadTensor *t, const double *grad_out);
void triad_tensor_zero_grad(TriadTensor *t);
void triad_ml_set_grad(int on);
int  triad_ml_get_grad(void);
void triad_ml_seed(uint64_t s);

TriadTensor *triad_tensor_add(TriadTensor *a, TriadTensor *b);
TriadTensor *triad_tensor_sub(TriadTensor *a, TriadTensor *b);
TriadTensor *triad_tensor_mul(TriadTensor *a, TriadTensor *b);
TriadTensor *triad_tensor_div(TriadTensor *a, TriadTensor *b);
TriadTensor *triad_tensor_neg(TriadTensor *a);
TriadTensor *triad_tensor_pow(TriadTensor *a, double exp_val);
TriadTensor *triad_tensor_matmul(TriadTensor *a, TriadTensor *b);
TriadTensor *triad_tensor_scale(TriadTensor *a, double s);
TriadTensor *triad_tensor_reshape(TriadTensor *a, int32_t ndim, const int32_t *shape);
TriadTensor *triad_tensor_flatten(TriadTensor *a);
TriadTensor *triad_tensor_transpose(TriadTensor *a, int32_t axis1, int32_t axis2);
TriadTensor *triad_tensor_bmm(TriadTensor *a, TriadTensor *b);
TriadTensor *triad_tensor_slice_axis(TriadTensor *a, int32_t axis,
                                     int32_t start, int32_t end, int32_t step);

TriadTensor *triad_tensor_eq(TriadTensor *a, TriadTensor *b);
TriadTensor *triad_tensor_ne(TriadTensor *a, TriadTensor *b);
TriadTensor *triad_tensor_lt(TriadTensor *a, TriadTensor *b);
TriadTensor *triad_tensor_le(TriadTensor *a, TriadTensor *b);
TriadTensor *triad_tensor_gt(TriadTensor *a, TriadTensor *b);
TriadTensor *triad_tensor_ge(TriadTensor *a, TriadTensor *b);

TriadTensor *triad_tensor_sum(TriadTensor *a);
TriadTensor *triad_tensor_mean(TriadTensor *a);
TriadTensor *triad_tensor_sum_axis(TriadTensor *a, int32_t axis, int keepdims);
TriadTensor *triad_tensor_mean_axis(TriadTensor *a, int32_t axis, int keepdims);
TriadTensor *triad_tensor_max_axis(TriadTensor *a, int32_t axis, int keepdims);
TriadTensor *triad_tensor_min_axis(TriadTensor *a, int32_t axis, int keepdims);

TriadTensor *triad_tensor_cat(TriadTensor **xs, int32_t n, int32_t axis);
TriadTensor *triad_tensor_stack(TriadTensor **xs, int32_t n, int32_t axis);

TriadTensor *triad_tensor_exp(TriadTensor *a);
TriadTensor *triad_tensor_log(TriadTensor *a);
TriadTensor *triad_tensor_sqrt(TriadTensor *a);
TriadTensor *triad_tensor_tanh(TriadTensor *a);
TriadTensor *triad_tensor_sigmoid(TriadTensor *a);
TriadTensor *triad_tensor_relu(TriadTensor *a);
TriadTensor *triad_tensor_dropout(TriadTensor *a, double p, int training);
TriadTensor *triad_tensor_softmax(TriadTensor *a);
TriadTensor *triad_tensor_softmax_axis(TriadTensor *a, int32_t axis);
TriadTensor *triad_tensor_layer_norm(TriadTensor *a, TriadTensor *gamma,
                                      TriadTensor *beta, double eps);

TriadTensor *triad_tensor_mse_loss(TriadTensor *pred, TriadTensor *target);
TriadTensor *triad_tensor_cross_entropy(TriadTensor *logits, TriadTensor *targets);
TriadTensor *triad_tensor_l1_loss(TriadTensor *pred, TriadTensor *target);
TriadTensor *triad_tensor_huber_loss(TriadTensor *pred, TriadTensor *target, double delta);
TriadTensor *triad_tensor_smooth_l1_loss(TriadTensor *pred, TriadTensor *target);
TriadTensor *triad_tensor_bce_loss(TriadTensor *pred, TriadTensor *target);
TriadTensor *triad_tensor_bce_with_logits(TriadTensor *logits, TriadTensor *target);
TriadTensor *triad_tensor_nll_loss(TriadTensor *log_probs, TriadTensor *targets);

typedef struct {
    TriadTensor *weight;
    TriadTensor *bias;
    int32_t in_features;
    int32_t out_features;
} Triadtriad;

Triadtriad *triad_triad_new(int32_t in_f, int32_t out_f, int use_bias);
void         triad_triad_free(Triadtriad *l);
TriadTensor *triad_triad_forward(Triadtriad *l, TriadTensor *x);

typedef struct {
    TriadTensor *weight;
    int32_t num_embeddings;
    int32_t embedding_dim;
} TriadEmbedding;

TriadEmbedding *triad_embedding_new(int32_t num_embeddings, int32_t embedding_dim);
void            triad_embedding_free(TriadEmbedding *e);
TriadTensor    *triad_embedding_forward(TriadEmbedding *e, TriadTensor *idx);

typedef struct {
    TriadTensor *gamma;
    TriadTensor *beta;
    int32_t normalized_shape;
    double eps;
} TriadLayerNorm;

TriadLayerNorm *triad_layer_norm_new(int32_t normalized_shape, double eps);
void            triad_layer_norm_free(TriadLayerNorm *ln);
TriadTensor    *triad_layer_norm_forward(TriadLayerNorm *ln, TriadTensor *x);

typedef struct {
    TriadTensor *gamma;
    TriadTensor *beta;
    double *running_mean;
    double *running_var;
    int32_t num_features;
    double eps;
    double momentum;
    int training;
} TriadBatchNorm1d;

TriadBatchNorm1d *triad_batch_norm1d_new(int32_t num_features, double eps, double momentum);
void              triad_batch_norm1d_free(TriadBatchNorm1d *bn);
TriadTensor      *triad_batch_norm1d_forward(TriadBatchNorm1d *bn, TriadTensor *x);
void              triad_batch_norm1d_train(TriadBatchNorm1d *bn, int training);

typedef enum {
    TRIAD_LAYER_triad,
    TRIAD_LAYER_RELU,
    TRIAD_LAYER_SIGMOID,
    TRIAD_LAYER_TANH,
    TRIAD_LAYER_SOFTMAX,
    TRIAD_LAYER_FLATTEN,
    TRIAD_LAYER_EMBEDDING,
    TRIAD_LAYER_LAYER_NORM,
} TriadLayerType;

typedef struct {
    TriadLayerType type;
    void *layer;
} TriadLayerEntry;

typedef struct {
    int32_t         nlayers;
    TriadLayerEntry *layers;
} TriadSequential;

TriadSequential *triad_sequential_new(int32_t nlayers);
void             triad_sequential_set(TriadSequential *s, int32_t i,
                                       TriadLayerType type, void *layer);
void             triad_sequential_free(TriadSequential *s);
TriadTensor     *triad_sequential_forward(TriadSequential *s, TriadTensor *x);

int32_t triad_sequential_params(TriadSequential *s, TriadTensor **out, int32_t max);

typedef struct {
    TriadTensor *weight;
    TriadTensor *bias;
    int32_t in_channels;
    int32_t out_channels;
    int32_t kernel_size;
    int32_t stride;
    int32_t padding;
} TriadConv1d;

TriadConv1d *triad_conv1d_new(int32_t in_c, int32_t out_c, int32_t ks,
                               int32_t stride, int32_t padding);
void         triad_conv1d_free(TriadConv1d *c);
TriadTensor *triad_conv1d_forward(TriadConv1d *c, TriadTensor *x);

typedef struct {
    TriadTensor *weight;
    TriadTensor *bias;
    int32_t in_channels;
    int32_t out_channels;
    int32_t kernel_size;
    int32_t stride;
    int32_t padding;
} TriadConv2d;

TriadConv2d *triad_conv2d_new(int32_t in_c, int32_t out_c, int32_t ks,
                               int32_t stride, int32_t padding);
void         triad_conv2d_free(TriadConv2d *c);
TriadTensor *triad_conv2d_forward(TriadConv2d *c, TriadTensor *x);

typedef struct {
    Triadtriad *fc1;
    Triadtriad *fc2;
    int32_t d_model;
    int32_t d_ff;
} TriadFeedForward;

TriadFeedForward *triad_feedforward_new(int32_t d_model, int32_t d_ff);
void              triad_feedforward_free(TriadFeedForward *ff);
TriadTensor      *triad_feedforward_forward(TriadFeedForward *ff, TriadTensor *x);
int32_t           triad_feedforward_params(TriadFeedForward *ff, TriadTensor **out, int32_t max);

typedef struct {
    Triadtriad *q_proj;
    Triadtriad *k_proj;
    Triadtriad *v_proj;
    Triadtriad *out_proj;
    int32_t d_model;
    int32_t n_heads;
} TriadMultiHeadAttention;

TriadMultiHeadAttention *triad_mha_new(int32_t d_model, int32_t n_heads);
void                     triad_mha_free(TriadMultiHeadAttention *m);
TriadTensor             *triad_mha_forward(TriadMultiHeadAttention *m, TriadTensor *x);
int32_t                  triad_mha_params(TriadMultiHeadAttention *m, TriadTensor **out, int32_t max);

typedef struct {
    TriadLayerNorm          *ln1;
    TriadMultiHeadAttention *attn;
    TriadLayerNorm          *ln2;
    TriadFeedForward        *ff;
    int32_t d_model;
} TriadTransformerBlock;

TriadTransformerBlock *triad_transformer_block_new(int32_t d_model, int32_t n_heads, int32_t d_ff);
void                   triad_transformer_block_free(TriadTransformerBlock *b);
TriadTensor           *triad_transformer_block_forward(TriadTransformerBlock *b, TriadTensor *x);
int32_t                triad_transformer_block_params(TriadTransformerBlock *b, TriadTensor **out, int32_t max);

typedef struct {
    TriadEmbedding         *embed;
    TriadTransformerBlock **blocks;
    TriadLayerNorm         *ln_final;
    int32_t n_blocks;
    int32_t d_model;
} TriadTransformer;

typedef struct {
    TriadTensor *wr, *wi;
    int32_t in_features, out_features;
} TriadWavetriad;

TriadWavetriad *triad_wave_triad_new(int32_t in_features, int32_t out_features);
void             triad_wave_triad_free(TriadWavetriad *w);
TriadTensor     *triad_wave_triad_forward(TriadWavetriad *w, TriadTensor *x);
int32_t          triad_wave_triad_params(TriadWavetriad *w, TriadTensor **out, int32_t max);

TriadTransformer *triad_transformer_new(int32_t vocab, int32_t d_model,
                                        int32_t n_blocks, int32_t n_heads, int32_t d_ff);
void              triad_transformer_free(TriadTransformer *t);
TriadTensor      *triad_transformer_forward(TriadTransformer *t, TriadTensor *idx);
int32_t           triad_transformer_params(TriadTransformer *t, TriadTensor **out, int32_t max);

typedef struct {
    int32_t      nparams;
    TriadTensor **params;
    double        lr;
    double        momentum;
    double      **velocity;
} TriadSGD;

TriadSGD *triad_sgd_new(TriadTensor **params, int32_t n, double lr, double momentum);
void      triad_sgd_step(TriadSGD *opt);
void      triad_sgd_zero_grad(TriadSGD *opt);
void      triad_sgd_free(TriadSGD *opt);

typedef struct {
    int32_t      nparams;
    TriadTensor **params;
    double        lr;
    double        beta1, beta2, eps;
    int32_t       t;
    double      **m;
    double      **v;
} TriadAdam;

TriadAdam *triad_adam_new(TriadTensor **params, int32_t n,
                           double lr, double beta1, double beta2, double eps);
void       triad_adam_step(TriadAdam *opt);
void       triad_adam_zero_grad(TriadAdam *opt);
void       triad_adam_free(TriadAdam *opt);

TriadTensor *triad_tensor_kl_div(TriadTensor *log_p, TriadTensor *q);
TriadTensor *triad_tensor_cosine_similarity_loss(TriadTensor *a, TriadTensor *b);

double triad_metric_accuracy(const TriadTensor *pred, const TriadTensor *target);
double triad_metric_mae(const TriadTensor *pred, const TriadTensor *target);
double triad_metric_rmse(const TriadTensor *pred, const TriadTensor *target);
double triad_metric_r2(const TriadTensor *pred, const TriadTensor *target);
double triad_metric_perplexity(double loss_value);
double triad_metric_top_k_accuracy(TriadTensor *pred, TriadTensor *target, int32_t k);
int32_t *triad_metric_confusion_matrix(TriadTensor *pred, TriadTensor *target, int32_t num_classes);
void triad_metric_precision_recall_f1(TriadTensor *pred, TriadTensor *target,
                                       int32_t num_classes,
                                       double *precision_out, double *recall_out, double *f1_out);

int  triad_save_weights(TriadTensor **params, int32_t n, const char *path);
int  triad_load_weights(TriadTensor **params, int32_t n, const char *path);
int  triad_save_checkpoint(TriadTensor **params, int32_t n, void *opt,
                            const char *opt_type, const char *path);
int  triad_load_checkpoint(TriadTensor **params, int32_t n, void *opt,
                            const char *opt_type, const char *path);

#endif
