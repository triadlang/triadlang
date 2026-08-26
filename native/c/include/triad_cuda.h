#ifndef TRIAD_CUDA_H
#define TRIAD_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int cuda_init(void);

void cuda_shutdown(void);

int cuda_matvec_f32(const float *W, const float *x, float *y,
                    int32_t rows, int32_t cols);

int cuda_matvec_f32_batched(const float *W, const float *x, float *y,
                             int32_t rows, int32_t cols, int32_t batch_size);

int cuda_dequant_matvec(int qtype,
                         const uint8_t *W_quant, const float *x, float *y,
                         int32_t rows, int32_t cols);

void cuda_dequant_reset(void);

#ifdef __cplusplus
}
#endif

#endif
