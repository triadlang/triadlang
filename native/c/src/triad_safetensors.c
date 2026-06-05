#include "triad_safetensors.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static uint64_t rd_le64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static char *st_strndup(const char *s, size_t n) {
    char *p = malloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = 0;
    return p;
}

static TriadSafeDType parse_dtype(const char *obj) {
    const char *p = strstr(obj, "\"dtype\"");
    if (!p) return TRIAD_ST_DTYPE_UNKNOWN;
    p = strchr(p, ':');
    if (!p) return TRIAD_ST_DTYPE_UNKNOWN;
    if (strstr(p, "\"BF16\"")) return TRIAD_ST_DTYPE_BF16;
    if (strstr(p, "\"F16\"")) return TRIAD_ST_DTYPE_F16;
    if (strstr(p, "\"F32\"")) return TRIAD_ST_DTYPE_F32;
    if (strstr(p, "\"F64\"")) return TRIAD_ST_DTYPE_F64;
    return TRIAD_ST_DTYPE_UNKNOWN;
}

static int parse_i64_array(const char *obj, const char *key,
                           int64_t *out, int32_t maxn) {
    const char *p = strstr(obj, key);
    if (!p) return -1;
    p = strchr(p, '[');
    if (!p) return -1;
    p++;
    int32_t n = 0;
    while (*p && *p != ']' && n < maxn) {
        while (*p == ' ' || *p == '\n' || *p == ',') p++;
        char *end = NULL;
        long long v = strtoll(p, &end, 10);
        if (end == p) break;
        out[n++] = (int64_t)v;
        p = end;
    }
    return n;
}

static int parse_u64_pair(const char *obj, const char *key,
                          uint64_t *a, uint64_t *b) {
    int64_t tmp[2] = {0, 0};
    int n = parse_i64_array(obj, key, tmp, 2);
    if (n != 2) return -1;
    *a = (uint64_t)tmp[0];
    *b = (uint64_t)tmp[1];
    return 0;
}

static int append_tensor(TriadSafeTensorFile *f, TriadSafeTensorInfo *info) {
    TriadSafeTensorInfo *next = realloc(
        f->tensors, (size_t)(f->ntensors + 1) * sizeof(TriadSafeTensorInfo));
    if (!next) return -1;
    f->tensors = next;
    f->tensors[f->ntensors++] = *info;
    return 0;
}

static int parse_header(TriadSafeTensorFile *f) {
    const char *p = f->header;
    while (*p) {
        while (*p && (*p == '{' || *p == ',' || *p == ' ' || *p == '\n')) p++;
        if (*p != '"') break;
        const char *name_start = ++p;
        while (*p && *p != '"') p++;
        if (*p != '"') break;
        char *name = st_strndup(name_start, (size_t)(p - name_start));
        if (!name) return -1;
        p++;
        while (*p && (*p == ' ' || *p == ':')) p++;
        if (*p != '{') {
            free(name);
            break;
        }

        const char *obj_start = p;
        int depth = 0;
        do {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            p++;
        } while (*p && depth > 0);
        const char *obj_end = p;
        char *obj = st_strndup(obj_start, (size_t)(obj_end - obj_start));
        if (!obj) {
            free(name);
            return -1;
        }

        if (strcmp(name, "__metadata__") != 0) {
            TriadSafeTensorInfo info;
            memset(&info, 0, sizeof(info));
            info.name = name;
            info.dtype = parse_dtype(obj);
            info.ndim = parse_i64_array(obj, "\"shape\"", info.shape, 8);
            if (info.ndim < 0 ||
                parse_u64_pair(obj, "\"data_offsets\"", &info.data_begin, &info.data_end) != 0) {
                free(info.name);
                free(obj);
                return -1;
            }
            if (append_tensor(f, &info) != 0) {
                free(info.name);
                free(obj);
                return -1;
            }
        } else {
            free(name);
        }
        free(obj);
    }
    return 0;
}

int triad_safetensors_open(const char *path, TriadSafeTensorFile *out) {
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return -1;
    }
    if (st.st_size < 16) {
        close(fd);
        return -1;
    }

    const uint8_t *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return -1;
    }

    uint64_t header_len = rd_le64(map);
    if (8 + header_len >= (uint64_t)st.st_size) {
        munmap((void*)map, (size_t)st.st_size);
        close(fd);
        return -1;
    }

    out->path = st_strndup(path, strlen(path));
    out->fd = fd;
    out->size = (uint64_t)st.st_size;
    out->header_len = header_len;
    out->data_base = 8 + header_len;
    out->map = map;
    out->header = st_strndup((const char*)map + 8, (size_t)header_len);
    if (!out->path || !out->header || parse_header(out) != 0) {
        triad_safetensors_close(out);
        return -1;
    }
    return 0;
}

void triad_safetensors_close(TriadSafeTensorFile *f) {
    if (!f) return;
    for (int32_t i = 0; i < f->ntensors; i++) free(f->tensors[i].name);
    free(f->tensors);
    free(f->header);
    free(f->path);
    if (f->map && f->map != MAP_FAILED) munmap((void*)f->map, (size_t)f->size);
    if (f->fd >= 0) close(f->fd);
    memset(f, 0, sizeof(*f));
    f->fd = -1;
}

const TriadSafeTensorInfo *triad_safetensors_find(const TriadSafeTensorFile *f,
                                                   const char *name) {
    for (int32_t i = 0; i < f->ntensors; i++)
        if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
    return NULL;
}

static double bf16_to_f64(uint16_t v) {
    union { uint32_t u; float f; } x;
    x.u = ((uint32_t)v) << 16;
    return (double)x.f;
}

static double f16_to_f64(uint16_t h) {
    uint16_t sign = (h >> 15) & 1;
    uint16_t exp = (h >> 10) & 0x1f;
    uint16_t frac = h & 0x3ff;
    double val;
    if (exp == 0) {
        val = frac ? ldexp((double)frac, -24) : 0.0;
    } else if (exp == 31) {
        val = frac ? NAN : INFINITY;
    } else {
        val = ldexp(1.0 + (double)frac / 1024.0, (int)exp - 15);
    }
    return sign ? -val : val;
}

int triad_safetensors_read_row_f64(const TriadSafeTensorFile *f,
                                   const TriadSafeTensorInfo *t,
                                   int64_t row, double *dst, int64_t dst_len) {
    if (!f || !t || t->ndim != 2 || row < 0 || row >= t->shape[0]) return -1;
    int64_t cols = t->shape[1];
    if (dst_len < cols) return -1;

    int elem_size = 0;
    if (t->dtype == TRIAD_ST_DTYPE_BF16 || t->dtype == TRIAD_ST_DTYPE_F16) elem_size = 2;
    else if (t->dtype == TRIAD_ST_DTYPE_F32) elem_size = 4;
    else if (t->dtype == TRIAD_ST_DTYPE_F64) elem_size = 8;
    else return -1;

    uint64_t offset = f->data_base + t->data_begin + (uint64_t)row * (uint64_t)cols * (uint64_t)elem_size;
    if (offset + (uint64_t)cols * (uint64_t)elem_size > f->data_base + t->data_end) return -1;
    const uint8_t *p = f->map + offset;
    for (int64_t i = 0; i < cols; i++) {
        if (t->dtype == TRIAD_ST_DTYPE_BF16) {
            uint16_t v;
            memcpy(&v, p + i * 2, 2);
            dst[i] = bf16_to_f64(v);
        } else if (t->dtype == TRIAD_ST_DTYPE_F16) {
            uint16_t v;
            memcpy(&v, p + i * 2, 2);
            dst[i] = f16_to_f64(v);
        } else if (t->dtype == TRIAD_ST_DTYPE_F32) {
            float v;
            memcpy(&v, p + i * 4, 4);
            dst[i] = (double)v;
        } else {
            double v;
            memcpy(&v, p + i * 8, 8);
            dst[i] = v;
        }
    }
    return (int)cols;
}

const char *triad_safetensors_dtype_name(TriadSafeDType dtype) {
    switch (dtype) {
        case TRIAD_ST_DTYPE_BF16: return "BF16";
        case TRIAD_ST_DTYPE_F16: return "F16";
        case TRIAD_ST_DTYPE_F32: return "F32";
        case TRIAD_ST_DTYPE_F64: return "F64";
        default: return "UNKNOWN";
    }
}
