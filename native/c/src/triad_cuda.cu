#include "triad_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t g_handle = NULL;
static int g_initialized = 0;
#define DEFAULT_BATCH_ROWS 65536



__device__ __forceinline__ float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) return 0.0f;
    float val = ldexpf((float)(mant + 1024), (int)exp - 25);
    return sign ? -val : val;
}

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t *q, 
                                                   uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}

__global__ void kernel_matvec_q4k(const uint8_t *W, const float *x, float *y,
                                   int32_t rows, int32_t cols) {
    extern __shared__ float sx[];
    if (threadIdx.x < cols) sx[threadIdx.x] = x[threadIdx.x];
    __syncthreads();
    int row = blockIdx.x;
    if (row >= rows) return;
    const int64_t nblk = cols / 256;
    const uint8_t *rp = W + (int64_t)row * nblk * 144;
    float sum = 0.0f;
    for (int64_t b = 0; b < nblk; b++) {
        const uint8_t *bp = rp + b * 144;
        uint16_t d16, m16; memcpy(&d16, bp, 2); memcpy(&m16, bp+2, 2);
        float d = f16_to_f32(d16), dm = f16_to_f32(m16);
        uint8_t sc[8], mn[8];
        for (int k = 0; k < 4; k++) {
            sc[k]=bp[4+k]&63; sc[k+4]=bp[8+k]&63;
            mn[k]=(bp[4+k]>>6)|((bp[12+k]&15)<<2); mn[k+4]=(bp[8+k]>>6)|((bp[12+k]>>4)<<2);
        }
        int bi = (int)(b * 256);
        for (int sb = 0; sb < 8; sb++) {
            float s = d * (float)sc[sb], mv = dm * (float)mn[sb];
            for (int j = 0; j < 16; j++) {
                uint8_t by = bp[16 + sb*16 + j];
                sum += (s*(by&15)-mv)*sx[bi+sb*32+j*2] + (s*(by>>4)-mv)*sx[bi+sb*32+j*2+1];
            }
        }
    }
    y[row] = sum;
}

__global__ void kernel_matvec_q5k(const uint8_t *W, const float *x, float *y,
                                   int32_t rows, int32_t cols) {
    extern __shared__ float sx[];
    if (threadIdx.x < cols) sx[threadIdx.x] = x[threadIdx.x];
    __syncthreads();
    int row = blockIdx.x;
    if (row >= rows) return;
    const int64_t nblk = cols / 256;
    const uint8_t *rp = W + (int64_t)row * nblk * 176;
    float sum = 0.0f;
    for (int64_t b = 0; b < nblk; b++) {
        const uint8_t *bp = rp + b * 176;
        uint16_t d16, m16; memcpy(&d16, bp, 2); memcpy(&m16, bp+2, 2);
        float d = f16_to_f32(d16), dm = f16_to_f32(m16);
        const uint8_t *sc = bp+4, *qh = bp+16, *ql = bp+48;
        uint8_t u1=1,u2=2; int is=0, bi=(int)(b*256);
        for (int n = 0; n < 256; n += 64) {
            uint8_t ds1,ms1,ds2,ms2;
            get_scale_min_k4(is,sc,&ds1,&ms1); get_scale_min_k4(is+1,sc,&ds2,&ms2);
            float s1=d*ds1,m1=dm*ms1,s2=d*ds2,m2=dm*ms2;
            for (int l = 0; l < 32; l++) {
                int v0 = (ql[l]&15)+((qh[l]&u1)?16:0);
                sum += (s1*v0-m1)*sx[bi+n+l];
            }
            for (int l = 0; l < 32; l++) {
                int v1 = (ql[l]>>4)+((qh[l]&u2)?16:0);
                sum += (s2*v1-m2)*sx[bi+n+l+32];
            }
            ql+=32; is+=2; u1<<=2; u2<<=2;
        }
    }
    y[row] = sum;
}

__global__ void kernel_matvec_q6k(const uint8_t *W, const float *x, float *y,
                                   int32_t rows, int32_t cols) {
    extern __shared__ float sx[];
    if (threadIdx.x < cols) sx[threadIdx.x] = x[threadIdx.x];
    __syncthreads();
    int row = blockIdx.x;
    if (row >= rows) return;
    const int64_t nblk = cols / 256;
    const uint8_t *rp = W + (int64_t)row * nblk * 210;
    float sum = 0.0f;
    for (int64_t b = 0; b < nblk; b++) {
        const uint8_t *bp = rp + b * 210;
        uint16_t d16; memcpy(&d16, bp+208, 2);
        float d = f16_to_f32(d16);
        const uint8_t *ql = bp, *qh = bp+128;
        const int8_t *sc = (const int8_t*)(bp+192);
        int bi = (int)(b*256);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l/16;
                int q1 = (int)((ql[l]&15)|((qh[l]>>0&3)<<4))-32;
                int q2 = (int)((ql[l+32]&15)|((qh[l]>>2&3)<<4))-32;
                int q3 = (int)((ql[l]>>4)|((qh[l]>>4&3)<<4))-32;
                int q4 = (int)((ql[l+32]>>4)|((qh[l]>>6&3)<<4))-32;
                sum += d*sc[is]*q1*sx[bi+n+l    ];
                sum += d*sc[is+2]*q2*sx[bi+n+l+32];
                sum += d*sc[is+4]*q3*sx[bi+n+l+64];
                sum += d*sc[is+6]*q4*sx[bi+n+l+96];
            }
            ql+=64; qh+=32; sc+=8;
        }
    }
    y[row] = sum;
}



extern "C" int cuda_init(void) {
    if (g_initialized) return 0;
    cudaError_t ce = cudaSetDevice(0);
    if (ce != cudaSuccess) { fprintf(stderr, "CUDA: no device (%s)\n", cudaGetErrorString(ce)); return -1; }
    cublasStatus_t cs = cublasCreate(&g_handle);
    if (cs != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "CUDA: cublasCreate failed\n"); return -1; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "CUDA: %s (%.1f GB, SM %d.%d)\n", prop.name, (float)prop.totalGlobalMem/(1024*1024*1024), prop.major, prop.minor);
    g_initialized = 1;
    return 0;
}

extern "C" void cuda_shutdown(void) {
    cuda_dequant_reset();
    if (g_handle) { cublasDestroy(g_handle); g_handle = NULL; }
    cudaDeviceReset();
    g_initialized = 0;
}



static int do_matvec(const float *W, const float *x, float *y,
                      int32_t rows, int32_t cols) {
    float *dW=NULL, *dx=NULL, *dy=NULL;
    cudaError_t ce; cublasStatus_t cs;
    float alpha=1.0f, beta=0.0f;
    size_t wb=(size_t)rows*cols*sizeof(float), xb=(size_t)cols*sizeof(float), yb=(size_t)rows*sizeof(float);
    ce = cudaMalloc(&dW, wb); if(ce!=cudaSuccess)goto fail;
    ce = cudaMalloc(&dx, xb); if(ce!=cudaSuccess)goto fail;
    ce = cudaMalloc(&dy, yb); if(ce!=cudaSuccess)goto fail;
    ce = cudaMemcpy(dW,W,wb,cudaMemcpyHostToDevice); if(ce!=cudaSuccess)goto fail;
    ce = cudaMemcpy(dx,x,xb,cudaMemcpyHostToDevice); if(ce!=cudaSuccess)goto fail;
    cs = cublasSgemv(g_handle, CUBLAS_OP_T, cols, rows, &alpha, dW, cols, dx, 1, &beta, dy, 1);
    if(cs!=CUBLAS_STATUS_SUCCESS)goto fail;
    ce = cudaMemcpy(y,dy,yb,cudaMemcpyDeviceToHost); if(ce!=cudaSuccess)goto fail;
    cudaFree(dW);cudaFree(dx);cudaFree(dy); return 0;
fail:
    if(dW)cudaFree(dW); if(dx)cudaFree(dx); if(dy)cudaFree(dy); return -1;
}

extern "C" int cuda_matvec_f32(const float *W, const float *x, float *y,
                     int32_t rows, int32_t cols) {
    if (!g_initialized) return -1;
    return do_matvec(W, x, y, rows, cols);
}

extern "C" int cuda_matvec_f32_batched(const float *W, const float *x, float *y,
                             int32_t rows, int32_t cols, int32_t batch_size) {
    if (!g_initialized) return -1;
    if (batch_size <= 0) batch_size = DEFAULT_BATCH_ROWS;
    if (batch_size > rows) batch_size = rows;
    float *dx=NULL, *dy=NULL; cudaError_t ce;
    ce = cudaMalloc(&dx, (size_t)cols*sizeof(float));
    if(ce!=cudaSuccess)return do_matvec(W,x,y,rows,cols);
    ce = cudaMemcpy(dx,x,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice);
    if(ce!=cudaSuccess){cudaFree(dx);return do_matvec(W,x,y,rows,cols);}
    size_t free_mem,total_mem; cudaMemGetInfo(&free_mem,&total_mem);
    size_t avail=(free_mem>256*1024*1024)?free_mem-256*1024*1024:free_mem/2;
    int32_t max_b=(int32_t)(avail/((size_t)cols*sizeof(float)));
    if(max_b<1)max_b=1; if(batch_size>max_b)batch_size=max_b;
    ce=cudaMalloc(&dy,(size_t)rows*sizeof(float));
    if(ce!=cudaSuccess){cudaFree(dx);return do_matvec(W,x,y,rows,cols);}
    int ok=0;
    for(int32_t off=0;off<rows;off+=batch_size){
        int32_t cur=(off+batch_size<=rows)?batch_size:(rows-off);
        float *dW=NULL; ce=cudaMalloc(&dW,(size_t)cur*cols*sizeof(float));
        if(ce!=cudaSuccess){ok=-1;break;}
        ce=cudaMemcpy(dW,W+(int64_t)off*cols,(size_t)cur*cols*sizeof(float),cudaMemcpyHostToDevice);
        if(ce!=cudaSuccess){cudaFree(dW);ok=-1;break;}
        float a=1,b=0; cublasStatus_t cs=cublasSgemv(g_handle,CUBLAS_OP_T,cols,cur,&a,dW,cols,dx,1,&b,dy+off,1);
        cudaFree(dW); if(cs!=CUBLAS_STATUS_SUCCESS){ok=-1;break;}
    }
    if(ok==0)cudaMemcpy(y,dy,(size_t)rows*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(dx);cudaFree(dy);
    if(ok==0)return 0;
    return do_matvec(W,x,y,rows,cols);
}



static uint8_t *g_dW = NULL; static size_t g_dWcap = 0;
static float *g_dx = NULL; static float *g_dy = NULL; static int32_t g_dy_rows = 0;

static int ensure_buf(size_t w, size_t x_s, int32_t yr) {
    if (w > g_dWcap) { if(g_dW)cudaFree(g_dW); g_dW=NULL; cudaError_t ce=cudaMalloc(&g_dW,w); if(ce!=cudaSuccess)return -1; g_dWcap=w; }
    if (!g_dx) { cudaError_t ce=cudaMalloc(&g_dx,x_s); if(ce!=cudaSuccess)return -1; }
    if (yr > g_dy_rows) { if(g_dy)cudaFree(g_dy); g_dy=NULL; cudaError_t ce=cudaMalloc(&g_dy,(size_t)yr*4); if(ce!=cudaSuccess)return -1; g_dy_rows=yr; }
    return 0;
}

extern "C" void cuda_dequant_reset(void) {
    if(g_dW){cudaFree(g_dW);g_dW=NULL;} if(g_dx){cudaFree(g_dx);g_dx=NULL;} if(g_dy){cudaFree(g_dy);g_dy=NULL;}
    g_dWcap=0; g_dy_rows=0;
}

extern "C" int cuda_dequant_matvec(int qtype,
                                    const uint8_t *W_quant, const float *x, float *y,
                                    int32_t rows, int32_t cols) {
    if (!g_initialized) return -1;
    size_t bb; if(qtype==12)bb=144; else if(qtype==13)bb=176; else if(qtype==14)bb=210; else return -1;
    int64_t bpr = cols / 256;
    size_t wn = (size_t)rows * bpr * bb, xn = (size_t)cols * 4;
    if(ensure_buf(wn, xn, rows)!=0)return -1;
    cudaMemcpy(g_dW, W_quant, wn, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dx, x, xn, cudaMemcpyHostToDevice);
    int blk = rows, thr = 256; size_t sm = (size_t)cols * 4;
    switch(qtype){
        case 12: kernel_matvec_q4k<<<blk,thr,sm>>>(g_dW,g_dx,g_dy,rows,cols); break;
        case 13: kernel_matvec_q5k<<<blk,thr,sm>>>(g_dW,g_dx,g_dy,rows,cols); break;
        case 14: kernel_matvec_q6k<<<blk,thr,sm>>>(g_dW,g_dx,g_dy,rows,cols); break;
        default: return -1;
    }
    if(cudaDeviceSynchronize()!=cudaSuccess)return -1;
    cudaMemcpy(y, g_dy, (size_t)rows*4, cudaMemcpyDeviceToHost);
    return 0;
}
