/* Direct smoke test for the native CUDA kernels (triad_cuda.cu):
 *  1. cuda_init / device probe
 *  2. cuda_matvec_f32         — exact 3x4 reference
 *  3. cuda_matvec_f32_batched — 64x256 vs CPU reference
 *  4. cuda_dequant_matvec     — Q6_K all-ones block (w=31 everywhere),
 *                               Q4_K/Q5_K zero blocks (y=0)
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

extern int  cuda_init(void);
extern void cuda_shutdown(void);
extern int  cuda_matvec_f32(const float *W, const float *x, float *y,
                            int32_t rows, int32_t cols);
extern int  cuda_matvec_f32_batched(const float *W, const float *x, float *y,
                                    int32_t rows, int32_t cols, int32_t batch);
extern int  cuda_dequant_matvec(int qtype, const uint8_t *W, const float *x,
                                float *y, int32_t rows, int32_t cols);

static int fails = 0;

static void check(const char *name, int cond) {
    printf("  %-28s %s\n", name, cond ? "PASS" : "FAIL");
    if (!cond) fails++;
}

static int close_enough(float a, float b) {
    float d = fabsf(a - b);
    return d < 1.0f || d < 0.001f * fabsf(b);
}

int main(void) {
    printf("[cuda-kernels] init\n");
    if (cuda_init() != 0) {
        printf("  cuda_init                  FAIL (no device)\n");
        return 1;
    }
    check("cuda_init", 1);

    /* 1) exact f32 matvec: W(3x4) row-major, x=[1,1,1,1] */
    {
        float W[12] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        float x[4]  = {1,1,1,1};
        float y[3]  = {0,0,0};
        float ref[3] = {10,26,42};
        int rc = cuda_matvec_f32(W, x, y, 3, 4);
        int ok = (rc == 0);
        for (int i = 0; i < 3; i++) ok = ok && close_enough(y[i], ref[i]);
        check("matvec_f32 3x4", ok);
        if (!ok) printf("    got y=[%f %f %f]\n", y[0], y[1], y[2]);
    }

    /* 2) batched f32 matvec vs CPU reference: 64x256 */
    {
        enum { R = 64, C = 256 };
        static float W[R * C], x[C], y[R], ref[R];
        for (int i = 0; i < R * C; i++) W[i] = (float)((i * 31 % 97) - 48) * 0.01f;
        for (int i = 0; i < C; i++) x[i] = (float)((i * 17 % 23) - 11) * 0.1f;
        for (int r = 0; r < R; r++) {
            float s = 0;
            for (int c = 0; c < C; c++) s += W[r * C + c] * x[c];
            ref[r] = s;
        }
        memset(y, 0, sizeof(y));
        int rc = cuda_matvec_f32_batched(W, x, y, R, C, 16);
        int ok = (rc == 0);
        for (int r = 0; r < R; r++) ok = ok && close_enough(y[r], ref[r]);
        check("matvec_f32_batched 64x256", ok);
    }

    /* 3) Q6_K all-ones: d=1.0, sc=1, ql/qh=0xFF -> every weight = 31 */
    {
        enum { R = 4, C = 256 };
        static uint8_t W[R * 210];
        float x[C], y[R];
        memset(W, 0xFF, sizeof(W));
        for (int r = 0; r < R; r++) {
            W[r * 210 + 192] = 1;          /* int8 scale = 1 (rest stay -1 -> see below) */
        }
        /* scales are int8 at bp+192..207; 0xFF = -1. Set all to +1: */
        for (int r = 0; r < R; r++)
            for (int i = 0; i < 16; i++) W[r * 210 + 192 + i] = 1;
        /* d (fp16 1.0 = 0x3C00) at bp+208 */
        for (int r = 0; r < R; r++) { W[r * 210 + 208] = 0x00; W[r * 210 + 209] = 0x3C; }
        float sx = 0;
        for (int i = 0; i < C; i++) { x[i] = (float)(i + 1); sx += x[i]; }
        float ref = 31.0f * sx;
        int rc = cuda_dequant_matvec(14, W, x, y, R, C);
        int ok = (rc == 0);
        for (int r = 0; r < R; r++) ok = ok && close_enough(y[r], ref);
        check("dequant_matvec Q6_K ones", ok);
        if (!ok) printf("    got y=[%f %f %f %f], ref %f\n", y[0], y[1], y[2], y[3], ref);
    }

    /* 4) Q4_K / Q5_K zero blocks -> y = 0 */
    {
        enum { R = 2, C = 256 };
        static uint8_t W4[R * 144], W5[R * 176];
        float x[C], y[R];
        for (int i = 0; i < C; i++) x[i] = 1.0f;
        int rc4 = cuda_dequant_matvec(12, W4, x, y, R, C);
        int ok4 = (rc4 == 0) && close_enough(y[0], 0) && close_enough(y[1], 0);
        check("dequant_matvec Q4_K zero", ok4);
        int rc5 = cuda_dequant_matvec(13, W5, x, y, R, C);
        int ok5 = (rc5 == 0) && close_enough(y[0], 0) && close_enough(y[1], 0);
        check("dequant_matvec Q5_K zero", ok5);
    }

    cuda_shutdown();
    printf("[cuda-kernels] %s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails ? 1 : 0;
}
