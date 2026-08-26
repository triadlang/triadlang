#ifndef TRIAD_GGUF_H
#define TRIAD_GGUF_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} GGUFValueType;

typedef enum {
    GGML_TYPE_F32   = 0,
    GGML_TYPE_F16   = 1,
    GGML_TYPE_Q4_0  = 2,
    GGML_TYPE_Q4_1  = 3,
    GGML_TYPE_Q5_0  = 6,
    GGML_TYPE_Q5_1  = 7,
    GGML_TYPE_Q8_0  = 8,
    GGML_TYPE_Q2_K  = 10,
    GGML_TYPE_Q3_K  = 11,
    GGML_TYPE_Q4_K  = 12,
    GGML_TYPE_Q5_K  = 13,
    GGML_TYPE_Q6_K  = 14,
    GGML_TYPE_Q8_K  = 15,
    GGML_TYPE_BF16  = 30,
} GGMLType;

typedef struct {
    char         *key;
    GGUFValueType type;
    union {
        uint8_t   u8;
        int8_t    i8;
        uint16_t  u16;
        int16_t   i16;
        uint32_t  u32;
        int32_t   i32;
        float     f32;
        uint8_t   bool_val;
        char     *str;
        uint64_t  u64;
        int64_t   i64;
        double    f64;
        struct {
            GGUFValueType elem_type;
            uint64_t      len;
            void         *data;
        } arr;
    } val;
} GGUFKeyValue;

typedef struct {
    char      *name;
    uint32_t   ndim;
    int64_t    shape[4];
    GGMLType   type;
    uint64_t   offset;

    int64_t    n_elements;
    size_t     n_bytes;
} GGUFTensorInfo;

typedef struct {
    uint32_t        version;
    int64_t         n_tensors;
    int64_t         n_kv;
    GGUFKeyValue   *kv;
    GGUFTensorInfo *tensors;

    uint64_t        data_offset;
    void           *mmap_addr;
    size_t          mmap_size;
    int             fd;
    char           *path;
} GGUFFile;

GGUFFile *gguf_open(const char *path);

void gguf_close(GGUFFile *f);

const GGUFKeyValue *gguf_find_kv(const GGUFFile *f, const char *key);

int64_t     gguf_get_int(const GGUFFile *f, const char *key, int64_t def);
double      gguf_get_float(const GGUFFile *f, const char *key, double def);
const char *gguf_get_str(const GGUFFile *f, const char *key, const char *def);

uint64_t gguf_get_array_len(const GGUFFile *f, const char *key, uint64_t def);
char    *gguf_get_array_str(const GGUFKeyValue *kv, uint64_t index);

const GGUFTensorInfo *gguf_find_tensor(const GGUFFile *f, const char *name);

const void *gguf_tensor_data(const GGUFFile *f, const GGUFTensorInfo *t);

double *gguf_dequantize(const GGUFFile *f, const GGUFTensorInfo *t);

int gguf_dequantize_row(const GGUFFile *f, const GGUFTensorInfo *t,
                        int64_t row, double *dst, int64_t dst_len);

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
} block_q4_K;

typedef struct {
    uint16_t d;
    int8_t   qs[32];
} block_q8_0;

typedef struct {
    uint16_t d;
    uint8_t  qs[16];
} block_q4_0;

void dequantize_q4_0(const void *src, double *dst, int64_t n_elements);
void dequantize_q4_K(const void *src, double *dst, int64_t n_elements);
void dequantize_q5_K(const void *src, double *dst, int64_t n_elements);
void dequantize_q6_K(const void *src, double *dst, int64_t n_elements);
void dequantize_q8_0(const void *src, double *dst, int64_t n_elements);
void dequantize_f16(const void *src, double *dst, int64_t n_elements);
void dequantize_f32(const void *src, double *dst, int64_t n_elements);

size_t  ggml_type_size(GGMLType t);
int64_t ggml_block_size(GGMLType t);

typedef void (*dequant_fn)(const void *src, double *dst, int64_t n_elements);

dequant_fn ggml_type_dequant_fn(GGMLType t);

#endif
