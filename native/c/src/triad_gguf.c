#define _POSIX_C_SOURCE 200809L
#include "triad_gguf.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static double f16_to_f64(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0 : 0.0;

        double v = ldexp((double)mant, -24);
        return sign ? -v : v;
    }
    if (exp == 31) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    double v = ldexp((double)(mant + 1024), (int)exp - 25);
    return sign ? -v : v;
}

typedef struct {
    const uint8_t *data;
    size_t         pos;
    size_t         size;
} Reader;

static uint8_t  r_u8(Reader *r)  { uint8_t v;  memcpy(&v, r->data + r->pos, 1); r->pos += 1; return v; }
static uint16_t r_u16(Reader *r) { uint16_t v; memcpy(&v, r->data + r->pos, 2); r->pos += 2; return v; }
static uint32_t r_u32(Reader *r) { uint32_t v; memcpy(&v, r->data + r->pos, 4); r->pos += 4; return v; }
static uint64_t r_u64(Reader *r) { uint64_t v; memcpy(&v, r->data + r->pos, 8); r->pos += 8; return v; }
static int32_t  r_i32(Reader *r) { int32_t v;  memcpy(&v, r->data + r->pos, 4); r->pos += 4; return v; }
static int64_t  r_i64(Reader *r) { int64_t v;  memcpy(&v, r->data + r->pos, 8); r->pos += 8; return v; }
static float    r_f32(Reader *r) { float v;    memcpy(&v, r->data + r->pos, 4); r->pos += 4; return v; }
static double   r_f64(Reader *r) { double v;   memcpy(&v, r->data + r->pos, 8); r->pos += 8; return v; }

static char *r_str(Reader *r) {
    uint64_t len = r_u64(r);
    char *s = malloc(len + 1);
    memcpy(s, r->data + r->pos, len);
    s[len] = '\0';
    r->pos += len;
    return s;
}

static void r_skip_val(Reader *r, GGUFValueType type);

static void r_skip_val(Reader *r, GGUFValueType type) {
    switch (type) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL: r->pos += 1; break;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: r->pos += 2; break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: r->pos += 4; break;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: r->pos += 8; break;
        case GGUF_TYPE_STRING: { uint64_t l = r_u64(r); r->pos += l; break; }
        case GGUF_TYPE_ARRAY: {
            uint32_t et = r_u32(r);
            uint64_t n = r_u64(r);
            for (uint64_t i = 0; i < n; i++) r_skip_val(r, (GGUFValueType)et);
            break;
        }
    }
}

static GGUFKeyValue r_kv(Reader *r) {
    GGUFKeyValue kv;
    kv.key = r_str(r);
    kv.type = (GGUFValueType)r_u32(r);
    switch (kv.type) {
        case GGUF_TYPE_UINT8:   kv.val.u8 = r_u8(r); break;
        case GGUF_TYPE_INT8:    kv.val.i8 = (int8_t)r_u8(r); break;
        case GGUF_TYPE_UINT16:  kv.val.u16 = r_u16(r); break;
        case GGUF_TYPE_INT16:   kv.val.i16 = (int16_t)r_u16(r); break;
        case GGUF_TYPE_UINT32:  kv.val.u32 = r_u32(r); break;
        case GGUF_TYPE_INT32:   kv.val.i32 = r_i32(r); break;
        case GGUF_TYPE_FLOAT32: kv.val.f32 = r_f32(r); break;
        case GGUF_TYPE_BOOL:    kv.val.bool_val = r_u8(r); break;
        case GGUF_TYPE_STRING:  kv.val.str = r_str(r); break;
        case GGUF_TYPE_UINT64:  kv.val.u64 = r_u64(r); break;
        case GGUF_TYPE_INT64:   kv.val.i64 = r_i64(r); break;
        case GGUF_TYPE_FLOAT64: kv.val.f64 = r_f64(r); break;
        case GGUF_TYPE_ARRAY: {
            kv.val.arr.elem_type = (GGUFValueType)r_u32(r);
            kv.val.arr.len = r_u64(r);

            kv.val.arr.data = (void*)(r->data + r->pos);
            for (uint64_t i = 0; i < kv.val.arr.len; i++)
                r_skip_val(r, kv.val.arr.elem_type);
            break;
        }
    }
    return kv;
}

size_t ggml_type_size(GGMLType t) {
    switch (t) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_Q4_0: return 18;
        case GGML_TYPE_Q4_1: return 20;
        case GGML_TYPE_Q5_0: return 22;
        case GGML_TYPE_Q5_1: return 24;
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q2_K: return 84;
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_Q8_K: return 292;
        default: return 0;
    }
}

int64_t ggml_block_size(GGMLType t) {
    switch (t) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16: return 1;
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q8_K: return 256;
        default: return 1;
    }
}

GGUFFile *gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("gguf_open"); return NULL; }

    struct stat st;
    fstat(fd, &st);
    size_t file_size = (size_t)st.st_size;

    void *addr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) { perror("mmap"); close(fd); return NULL; }

    Reader r = { .data = (const uint8_t*)addr, .pos = 0, .size = file_size };

    uint32_t magic = r_u32(&r);
    if (magic != 0x46554747) {
        fprintf(stderr, "not a GGUF file (magic: 0x%08x)\n", magic);
        munmap(addr, file_size);
        close(fd);
        return NULL;
    }

    GGUFFile *f = calloc(1, sizeof(GGUFFile));
    f->path = strdup(path);
    f->mmap_addr = addr;
    f->mmap_size = file_size;
    f->fd = fd;

    f->version = r_u32(&r);
    f->n_tensors = (int64_t)r_u64(&r);
    f->n_kv = (int64_t)r_u64(&r);

    f->kv = calloc(f->n_kv, sizeof(GGUFKeyValue));
    for (int64_t i = 0; i < f->n_kv; i++)
        f->kv[i] = r_kv(&r);

    f->tensors = calloc(f->n_tensors, sizeof(GGUFTensorInfo));
    for (int64_t i = 0; i < f->n_tensors; i++) {
        GGUFTensorInfo *ti = &f->tensors[i];
        ti->name = r_str(&r);
        ti->ndim = r_u32(&r);
        ti->n_elements = 1;
        for (uint32_t d = 0; d < ti->ndim; d++) {
            ti->shape[d] = (int64_t)r_u64(&r);
            ti->n_elements *= ti->shape[d];
        }
        ti->type = (GGMLType)r_u32(&r);
        ti->offset = r_u64(&r);

        int64_t bs = ggml_block_size(ti->type);
        size_t ts = ggml_type_size(ti->type);
        ti->n_bytes = (size_t)((ti->n_elements + bs - 1) / bs) * ts;
    }

    size_t align = 32;
    f->data_offset = (r.pos + align - 1) & ~(align - 1);

    return f;
}

void gguf_close(GGUFFile *f) {
    if (!f) return;
    for (int64_t i = 0; i < f->n_kv; i++) {
        free(f->kv[i].key);
        if (f->kv[i].type == GGUF_TYPE_STRING) free(f->kv[i].val.str);
    }
    free(f->kv);
    for (int64_t i = 0; i < f->n_tensors; i++)
        free(f->tensors[i].name);
    free(f->tensors);
    if (f->mmap_addr) munmap(f->mmap_addr, f->mmap_size);
    if (f->fd >= 0) close(f->fd);
    free(f->path);
    free(f);
}

const GGUFKeyValue *gguf_find_kv(const GGUFFile *f, const char *key) {
    for (int64_t i = 0; i < f->n_kv; i++)
        if (strcmp(f->kv[i].key, key) == 0) return &f->kv[i];
    return NULL;
}

int64_t gguf_get_int(const GGUFFile *f, const char *key, int64_t def) {
    const GGUFKeyValue *kv = gguf_find_kv(f, key);
    if (!kv) return def;
    switch (kv->type) {
        case GGUF_TYPE_UINT32: return kv->val.u32;
        case GGUF_TYPE_INT32:  return kv->val.i32;
        case GGUF_TYPE_UINT64: return (int64_t)kv->val.u64;
        case GGUF_TYPE_INT64:  return kv->val.i64;
        case GGUF_TYPE_UINT8:  return kv->val.u8;
        case GGUF_TYPE_INT8:   return kv->val.i8;
        default: return def;
    }
}

double gguf_get_float(const GGUFFile *f, const char *key, double def) {
    const GGUFKeyValue *kv = gguf_find_kv(f, key);
    if (!kv) return def;
    if (kv->type == GGUF_TYPE_FLOAT32) return kv->val.f32;
    if (kv->type == GGUF_TYPE_FLOAT64) return kv->val.f64;
    return def;
}

const char *gguf_get_str(const GGUFFile *f, const char *key, const char *def) {
    const GGUFKeyValue *kv = gguf_find_kv(f, key);
    if (!kv || kv->type != GGUF_TYPE_STRING) return def;
    return kv->val.str;
}

uint64_t gguf_get_array_len(const GGUFFile *f, const char *key, uint64_t def) {
    const GGUFKeyValue *kv = gguf_find_kv(f, key);
    if (!kv || kv->type != GGUF_TYPE_ARRAY) return def;
    return kv->val.arr.len;
}

char *gguf_get_array_str(const GGUFKeyValue *kv, uint64_t index) {
    if (!kv || kv->type != GGUF_TYPE_ARRAY ||
        kv->val.arr.elem_type != GGUF_TYPE_STRING ||
        index >= kv->val.arr.len) {
        return NULL;
    }

    const uint8_t *p = (const uint8_t*)kv->val.arr.data;
    for (uint64_t i = 0; i < kv->val.arr.len; i++) {
        uint64_t slen;
        memcpy(&slen, p, 8);
        p += 8;
        if (i == index) {
            char *s = malloc((size_t)slen + 1);
            if (!s) return NULL;
            memcpy(s, p, (size_t)slen);
            s[slen] = '\0';
            return s;
        }
        p += slen;
    }
    return NULL;
}

const GGUFTensorInfo *gguf_find_tensor(const GGUFFile *f, const char *name) {
    for (int64_t i = 0; i < f->n_tensors; i++)
        if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
    return NULL;
}

const void *gguf_tensor_data(const GGUFFile *f, const GGUFTensorInfo *t) {
    return (const uint8_t*)f->mmap_addr + f->data_offset + t->offset;
}

void dequantize_f32(const void *src, double *dst, int64_t n) {
    const float *s = (const float*)src;
    for (int64_t i = 0; i < n; i++) dst[i] = (double)s[i];
}

void dequantize_f16(const void *src, double *dst, int64_t n) {
    const uint16_t *s = (const uint16_t*)src;
    for (int64_t i = 0; i < n; i++) dst[i] = f16_to_f64(s[i]);
}

void dequantize_q8_0(const void *src, double *dst, int64_t n) {
    const block_q8_0 *blocks = (const block_q8_0*)src;
    int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; b++) {
        double d = f16_to_f64(blocks[b].d);
        for (int j = 0; j < 32; j++)
            dst[b * 32 + j] = d * (double)blocks[b].qs[j];
    }
}

void dequantize_q4_0(const void *src, double *dst, int64_t n) {
    const block_q4_0 *blocks = (const block_q4_0*)src;
    int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; b++) {
        double d = f16_to_f64(blocks[b].d);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blocks[b].qs[j];
            dst[b * 32 + j * 2]     = d * (double)((int)(byte & 0xF) - 8);
            dst[b * 32 + j * 2 + 1] = d * (double)((int)(byte >> 4) - 8);
        }
    }
}

void dequantize_q4_K(const void *src, double *dst, int64_t n) {
    const block_q4_K *blocks = (const block_q4_K*)src;
    int64_t nb = n / 256;
    for (int64_t b = 0; b < nb; b++) {
        double d = f16_to_f64(blocks[b].d);
        double dmin = f16_to_f64(blocks[b].dmin);

        uint8_t sc[8], mn[8];
        for (int i = 0; i < 4; i++) {
            sc[i]     = blocks[b].scales[i] & 0x3F;
            sc[i + 4] = blocks[b].scales[i + 4] & 0x3F;
            mn[i]     = blocks[b].scales[i] >> 6 | ((blocks[b].scales[i + 8] & 0x0F) << 2);
            mn[i + 4] = blocks[b].scales[i + 4] >> 6 | ((blocks[b].scales[i + 8] >> 4) << 2);
        }

        for (int sb = 0; sb < 8; sb++) {
            double scale = d * (double)sc[sb];
            double min_val = dmin * (double)mn[sb];
            int off = sb * 32;
            for (int j = 0; j < 16; j++) {
                uint8_t byte = blocks[b].qs[sb * 16 + j];
                dst[b * 256 + off + j * 2]     = scale * (double)(byte & 0xF) - min_val;
                dst[b * 256 + off + j * 2 + 1] = scale * (double)(byte >> 4) - min_val;
            }
        }
    }
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

void dequantize_q5_K(const void *src, double *dst, int64_t n) {
    const uint8_t *blocks = (const uint8_t*)src;
    int64_t nb = n / 256;
    size_t bsz = 176;

    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *bp = blocks + i * bsz;
        uint16_t d_f16, dmin_f16;
        memcpy(&d_f16, bp, 2);
        memcpy(&dmin_f16, bp + 2, 2);
        double d = f16_to_f64(d_f16);
        double dmin = f16_to_f64(dmin_f16);

        const uint8_t *scales = bp + 4;
        const uint8_t *qh = bp + 16;
        const uint8_t *ql = bp + 48;

        uint8_t u1 = 1, u2 = 2;
        int is = 0;
        int out_idx = (int)(i * 256);

        for (int j = 0; j < 256; j += 64) {
            uint8_t d_sc, m_sc;
            get_scale_min_k4(is + 0, scales, &d_sc, &m_sc);
            double d1 = d * (double)d_sc;
            double m1 = dmin * (double)m_sc;
            get_scale_min_k4(is + 1, scales, &d_sc, &m_sc);
            double d2 = d * (double)d_sc;
            double m2 = dmin * (double)m_sc;

            for (int l = 0; l < 32; l++) {
                int v0 = (ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                dst[out_idx++] = d1 * (double)v0 - m1;
            }
            for (int l = 0; l < 32; l++) {
                int v1 = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                dst[out_idx++] = d2 * (double)v1 - m2;
            }

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void dequantize_q6_K(const void *src, double *dst, int64_t n) {
    const uint8_t *blocks = (const uint8_t*)src;
    int64_t nb = n / 256;
    size_t bsz = 210;

    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *bp = blocks + i * bsz;
        uint16_t d_f16;
        memcpy(&d_f16, bp + 208, 2);
        double d = f16_to_f64(d_f16);

        const uint8_t *ql = bp;
        const uint8_t *qh = bp + 128;
        const int8_t *sc = (const int8_t*)(bp + 192);
        int base = (int)(i * 256);

        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int o0 = base + n + l;
                int o1 = base + n + l + 32;
                int o2 = base + n + l + 64;
                int o3 = base + n + l + 96;

                int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                dst[o0] = d * (double)sc[is + 0] * (double)q1;
                dst[o1] = d * (double)sc[is + 2] * (double)q2;
                dst[o2] = d * (double)sc[is + 4] * (double)q3;
                dst[o3] = d * (double)sc[is + 6] * (double)q4;
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

dequant_fn ggml_type_dequant_fn(GGMLType t) {
    switch (t) {
        case GGML_TYPE_F32:  return dequantize_f32;
        case GGML_TYPE_F16:  return dequantize_f16;
        case GGML_TYPE_Q4_0: return dequantize_q4_0;
        case GGML_TYPE_Q4_K: return dequantize_q4_K;
        case GGML_TYPE_Q5_K: return dequantize_q5_K;
        case GGML_TYPE_Q6_K: return dequantize_q6_K;
        case GGML_TYPE_Q8_0: return dequantize_q8_0;
        default: return NULL;
    }
}

double *gguf_dequantize(const GGUFFile *f, const GGUFTensorInfo *t) {
    const void *raw = gguf_tensor_data(f, t);
    double *out = malloc(t->n_elements * sizeof(double));

    switch (t->type) {
        case GGML_TYPE_F32:  dequantize_f32(raw, out, t->n_elements); break;
        case GGML_TYPE_F16:  dequantize_f16(raw, out, t->n_elements); break;
        case GGML_TYPE_Q4_0: dequantize_q4_0(raw, out, t->n_elements); break;
        case GGML_TYPE_Q4_K: dequantize_q4_K(raw, out, t->n_elements); break;
        case GGML_TYPE_Q5_K: dequantize_q5_K(raw, out, t->n_elements); break;
        case GGML_TYPE_Q6_K: dequantize_q6_K(raw, out, t->n_elements); break;
        case GGML_TYPE_Q8_0: dequantize_q8_0(raw, out, t->n_elements); break;
        default:
            fprintf(stderr, "unsupported type %d for tensor %s\n", t->type, t->name);
            memset(out, 0, t->n_elements * sizeof(double));
            break;
    }
    return out;
}

int gguf_dequantize_row(const GGUFFile *f, const GGUFTensorInfo *t,
                        int64_t row, double *dst, int64_t dst_len) {
    if (!f || !t || !dst || t->ndim != 2) return -1;
    int64_t cols = t->shape[0];
    int64_t rows = t->shape[1];
    if (row < 0 || row >= rows || dst_len < cols) return -1;

    int64_t bs = ggml_block_size(t->type);
    size_t ts = ggml_type_size(t->type);
    dequant_fn fn = ggml_type_dequant_fn(t->type);
    if (!fn || bs <= 0 || ts == 0) return -1;

    int64_t blocks_per_row = (cols + bs - 1) / bs;
    size_t row_bytes = (size_t)blocks_per_row * ts;
    const uint8_t *raw = (const uint8_t*)gguf_tensor_data(f, t);
    const void *row_ptr = raw + (size_t)row * row_bytes;

    fn(row_ptr, dst, cols);
    return 0;
}
