#ifndef TRIAD_CUDA_H
#define TRIAD_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize CUDA device + cuBLAS handle. Returns 0 on success. */
int cuda_init(void);

/* Shutdown: destroy handle, reset device */
void cuda_shutdown(void);

/* matvec: y[rows] = W(rows,cols) * x(cols), all f32 host arrays.
   W is row-major: W[i*cols + j] is element (i,j).
   Uses cuBLAS internally (copies to/from GPU). */
int cuda_matvec_f32(const float *W, const float *x, float *y,
                    int32_t rows, int32_t cols);

/* Version that splits into row batches (for large tensors that
   might not fit in GPU memory). batch_size = 0 uses auto-chunk. */
int cuda_matvec_f32_batched(const float *W, const float *x, float *y,
                             int32_t rows, int32_t cols, int32_t batch_size);

/* Fused dequant + matvec for quantized weights directly on GPU.
   qtype uses GGML ids: 12=Q4_K, 13=Q5_K, 14=Q6_K
   W_quant is the raw quantized data (same layout as CPU dequant).
   Avoids materializing f32 weights on host.
   Returns 0 on success, -1 on error. */
int cuda_dequant_matvec(int qtype,
                         const uint8_t *W_quant, const float *x, float *y,
                         int32_t rows, int32_t cols);

/* Free persistent GPU buffers */
void cuda_dequant_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* TRIAD_CUDA_H */
