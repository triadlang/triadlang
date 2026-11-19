#ifndef TRIAD_SAFETENSORS_H
#define TRIAD_SAFETENSORS_H

#include <stdint.h>

typedef enum {
    TRIAD_ST_DTYPE_UNKNOWN = 0,
    TRIAD_ST_DTYPE_BF16,
    TRIAD_ST_DTYPE_F16,
    TRIAD_ST_DTYPE_F32,
    TRIAD_ST_DTYPE_F64,
} TriadSafeDType;

typedef struct {
    char *name;
    TriadSafeDType dtype;
    int32_t ndim;
    int64_t shape[8];
    uint64_t data_begin;
    uint64_t data_end;
} TriadSafeTensorInfo;

typedef struct {
    char *path;
    int fd;
    uint64_t size;
    uint64_t header_len;
    uint64_t data_base;
    const uint8_t *map;
    char *header;
    int32_t ntensors;
    TriadSafeTensorInfo *tensors;
} TriadSafeTensorFile;

int  triad_safetensors_open(const char *path, TriadSafeTensorFile *out);
void triad_safetensors_close(TriadSafeTensorFile *f);

const TriadSafeTensorInfo *triad_safetensors_find(const TriadSafeTensorFile *f,
                                                   const char *name);
int triad_safetensors_read_row_f64(const TriadSafeTensorFile *f,
                                   const TriadSafeTensorInfo *t,
                                   int64_t row, double *dst, int64_t dst_len);

const char *triad_safetensors_dtype_name(TriadSafeDType dtype);

#endif /* TRIAD_SAFETENSORS_H */
