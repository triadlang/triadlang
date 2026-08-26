#ifndef TRIAD_KERNEL_RT_H
#define TRIAD_KERNEL_RT_H

// Versão do triad_rt.h adaptada para o kernel (sem dependências externas)
// Usada pelo motor quântico interno

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// No kernel, usamos alocação simples (sem GC)
#define TRIAD_NO_BOEHM 1

// Complex number type
typedef struct {
    double re;
    double im;
} TriadCplx;

// Basic types for kernel
typedef struct TriadString TriadString;
typedef struct TriadList TriadList;
typedef struct TriadDict TriadDict;

struct TriadString {
    int32_t refcount;
    int32_t len;
    int32_t cap;
    char *data;
};

struct TriadList {
    int32_t refcount;
    int32_t len;
    int32_t cap;
    void **items;
};

struct TriadDict {
    int32_t refcount;
    int32_t len;
    int32_t cap;
    char **keys;
    void **vals;
};

// Math constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

// Simple math functions for kernel
static inline double triad_sqrt(double x) {
    if (x < 0) return 0;
    double y = x / 2.0;
    for (int i = 0; i < 20; i++) {
        y = (y + x / y) / 2.0;
    }
    return y;
}

static inline double triad_exp(double x) {
    double result = 1.0;
    double term = 1.0;
    for (int i = 1; i < 50; i++) {
        term *= x / i;
        result += term;
        if (term < 1e-15) break;
    }
    return result;
}

static inline double triad_log(double x) {
    if (x <= 0) return -1e30;
    double y = x / 2.0;
    for (int i = 0; i < 50; i++) {
        y = y - (triad_exp(y) - x) / triad_exp(y);
    }
    return y;
}

static inline double triad_sin(double x) {
    double result = x;
    double term = x;
    for (int i = 1; i < 30; i++) {
        term *= -x * x / ((2.0 * i) * (2.0 * i + 1.0));
        result += term;
        if (term < 1e-15 && term > -1e-15) break;
    }
    return result;
}

static inline double triad_cos(double x) {
    return triad_sin(x + M_PI / 2.0);
}

static inline double triad_pow(double base, double exp) {
    if (exp == 0) return 1.0;
    if (base == 0) return 0.0;
    return triad_exp(exp * triad_log(base));
}

static inline double triad_fabs(double x) {
    return x < 0 ? -x : x;
}

static inline double triad_atan(double x);
static inline double triad_atan2(double y, double x);

// Memory functions (kernel uses its own heap)
extern void *heap_alloc(size_t size);
extern void heap_free(void *ptr);

static inline void *triad_kmalloc(size_t size) {
    return heap_alloc(size);
}

static inline void triad_kfree(void *ptr) {
    heap_free(ptr);
}

// String functions for kernel
static inline int triad_strlen(const char *s) {
    int len = 0;
    while (s[len]) len++;
    return len;
}

static inline char *triad_strdup(const char *s) {
    int len = triad_strlen(s);
    char *copy = (char *)triad_kmalloc(len + 1);
    for (int i = 0; i <= len; i++) {
        copy[i] = s[i];
    }
    return copy;
}

static inline int triad_strcmp(const char *a, const char *b) {
    while (*a && *b && *a == *b) {
        a++;
        b++;
    }
    return (int)*a - (int)*b;
}

static inline int triad_strncmp(const char *a, const char *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return (int)a[i] - (int)b[i];
        if (!a[i]) return 0;
    }
    return 0;
}

static inline void triad_memcpy(void *dst, const void *src, size_t n) {
    char *d = (char *)dst;
    const char *s = (const char *)src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
}

static inline void triad_memset(void *dst, int c, size_t n) {
    char *d = (char *)dst;
    for (size_t i = 0; i < n; i++) {
        d[i] = (char)c;
    }
}

// Print function (kernel uses VGA)
extern void vga_puts(const char *s);
extern void vga_putc(char c);

static inline void triad_kprint(const char *s) {
    vga_puts(s);
}

// Simple random number generator for kernel
static inline uint64_t triad_rand64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline double triad_randn(uint64_t *state) {
    double u1 = (double)(triad_rand64(state) >> 11) / (double)(1ULL << 53);
    double u2 = (double)(triad_rand64(state) >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    return triad_sqrt(-2.0 * triad_log(u1)) * triad_cos(2.0 * M_PI * u2);
}

// TriadLang value types for kernel interpreter
typedef enum {
    TRIAD_KERNEL_NONE = 0,
    TRIAD_KERNEL_INT,
    TRIAD_KERNEL_FLOAT,
    TRIAD_KERNEL_STRING,
    TRIAD_KERNEL_BOOL,
    TRIAD_KERNEL_LIST,
    TRIAD_KERNEL_DICT,
} TriadKernelTag;

typedef struct {
    TriadKernelTag tag;
    union {
        int64_t ival;
        double fval;
        char sval[256];
        bool bval;
        struct {
            void **items;
            int count;
        } list;
        struct {
            char **keys;
            void **vals;
            int count;
        } dict;
    };
} TriadKernelValue;

// Forward declarations for quantum kernel
struct QosProcess;
struct QosRuntime;

// Kernel quantum functions
void qos_kernel_init_process(struct QosProcess *p, int N, double L);
void qos_kernel_evolve_all(struct QosRuntime *rt, double T);

#endif