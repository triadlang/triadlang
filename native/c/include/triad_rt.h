#ifndef TRIAD_RT_H
#define TRIAD_RT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#ifndef TRIAD_NO_BOEHM
  #include <gc/gc.h>

  #undef malloc
  #undef realloc
  #undef free
  #undef calloc

  #define malloc(sz)          GC_MALLOC(sz)
  #define calloc(n, sz)       GC_MALLOC((n) * (sz))
  #define realloc(p, sz)      GC_REALLOC(p, sz)
  #define free(p)             GC_FREE(p)

  #define TRIAD_GC_BOEHM 1
#else
  #define TRIAD_GC_BOEHM 0
#endif

void triad_runtime_init(void);

#define TriadMain()                                     \
    int main(int argc, char **argv) {                   \
        (void)argc; (void)argv;                         \
        triad_runtime_init();

#define TriadMainEnd()                                  \
        return 0;                                       \
    }

typedef struct TriadString  TriadString;
typedef struct TriadList    TriadList;
typedef struct TriadDict    TriadDict;
typedef struct TriadDictEntry TriadDictEntry;
typedef struct TriadNDArray TriadNDArray;
typedef struct TriadClosure TriadClosure;
typedef struct TriadIter    TriadIter;
typedef struct TriadObject  TriadObject;
typedef struct TriadTuple   TriadTuple;
typedef struct TriadBytes   TriadBytes;
typedef struct TriadComplex TriadComplex;
typedef struct TriadSet     TriadSet;
typedef struct TriadGenerator TriadGenerator;
typedef struct TriadRegex TriadRegex;

typedef enum {
    TRIAD_NONE = 0,
    TRIAD_BOOL,
    TRIAD_INT,
    TRIAD_FLOAT,
    TRIAD_STRING,
    TRIAD_LIST,
    TRIAD_DICT,
    TRIAD_NDARRAY,
    TRIAD_CLOSURE,
    TRIAD_ITER,
    TRIAD_OBJECT,
    TRIAD_NATIVE_FN,
    TRIAD_PTR,
    TRIAD_PYOBJ,
    TRIAD_TUPLE,
    TRIAD_BYTES,
    TRIAD_COMPLEX,
    TRIAD_SET,
    TRIAD_GENERATOR,
    TRIAD_REGEX,
} TriadTag;

struct TriadString {
    int32_t  refcount;
    int32_t  len;
    int32_t  cap;
    char    *data;
};

typedef struct TriadValue TriadValue;
typedef TriadValue (*TriadNativeFn)(int nargs, TriadValue *args);

struct TriadValue {
    TriadTag tag;
    union {
        bool           bval;
        int64_t        ival;
        double         fval;
        TriadString   *sval;
        TriadList     *lval;
        TriadDict     *dval;
        TriadNDArray  *aval;
        TriadClosure  *cval;
        TriadIter     *itval;
        TriadObject   *oval;
        TriadNativeFn  nfn;
        void          *ptr;
        TriadTuple    *tval;
        TriadBytes    *bvalp;
        TriadComplex  *cvalp;
        TriadSet      *sset;
        TriadGenerator *gval;
        TriadRegex *rxval;
    } as;
};

#define TRIAD_NONE_VAL    ((TriadValue){.tag = TRIAD_NONE, .as = {.ival = 0}})
#define TRIAD_BOOL(v)     ((TriadValue){.tag = TRIAD_BOOL, .as = {.bval = (v)}})
#define TRIAD_INT(v)      ((TriadValue){.tag = TRIAD_INT, .as = {.ival = (v)}})
#define TRIAD_FLOAT(v)    ((TriadValue){.tag = TRIAD_FLOAT, .as = {.fval = (v)}})
#define TRIAD_PTR_VAL(p)  ((TriadValue){.tag = TRIAD_PTR, .as = {.ptr = (void*)(p)}})
#define TRIAD_PYOBJ_VAL(p) ((TriadValue){.tag = TRIAD_PYOBJ, .as = {.ptr = (void*)(p)}})
#define TRIAD_REGEX_VAL(p) ((TriadValue){.tag = TRIAD_REGEX, .as = {.rxval = (p)}})

extern TriadString *(*triad_pyobj_str_hook)(void *pyobj);
extern bool (*triad_pyobj_y_hook)(void *pyobj);

struct TriadList {
    int32_t     refcount;
    int32_t     len;
    int32_t     cap;
    TriadValue *items;
};

struct TriadDictEntry {
    TriadString *key;
    TriadValue   value;
    int32_t      hash;
    int32_t      seq;
};

struct TriadDict {
    int32_t          refcount;
    int32_t          len;
    int32_t          cap;
    int32_t          tombstones;
    int32_t          next_seq;
    TriadDictEntry  *entries;
};

struct TriadNDArray {
    int32_t     refcount;
    int32_t     ndim;
    int32_t    *shape;
    int32_t    *strides;
    int64_t     size;
    double     *data;
    bool        owns_data;
};

struct TriadClosure {
    int32_t         refcount;
    TriadNativeFn   fn;
    int32_t         ncaptured;
    TriadValue     *captured;
    void           *user_data;
};

typedef TriadValue (*TriadIterNextFn)(struct TriadIter *it);

struct TriadIter {
    int32_t          refcount;
    int32_t          state;
    TriadIterNextFn  next_fn;
    TriadValue       current;
    void            *user_data;
};

struct TriadObject {
    int32_t     refcount;
    TriadString *type_name;
    TriadDict   *fields;
    struct TriadClassMeta *class_meta;
};

typedef struct TriadClassMethod {
    const char        *name;
    TriadNativeFn      fn;
} TriadClassMethod;

typedef struct TriadClassMeta TriadClassMeta;

struct TriadClassMeta {
    const char          *name;
    TriadClassMeta      *parent;
    TriadClassMethod    *methods;
    int32_t              methods_len;
    const char         **fields;
    int32_t              fields_len;
    int32_t              refcount;
};

struct TriadTuple {
    int32_t     refcount;
    int32_t     len;
    TriadValue *items;
};

struct TriadBytes {
    int32_t     refcount;
    int32_t     len;
    uint8_t    *data;
};

struct TriadComplex {
    int32_t     refcount;
    double      re;
    double      im;
};

typedef struct TriadSetEntry TriadSetEntry;

struct TriadSet {
    int32_t          refcount;
    int32_t          len;
    int32_t          cap;
    TriadSetEntry   *entries;
};

struct TriadSetEntry {
    TriadValue   key;
    int32_t      hash;
    bool         used;
};

struct TriadGenerator {
    int32_t         refcount;
    int32_t         state;
    TriadValue      current;
    TriadValue     (*next_fn)(struct TriadGenerator *g);
    void           *user_data;
};

TriadTuple *triad_tuple_new(int32_t len);
void        triad_tuple_free(TriadTuple *t);
TriadValue  triad_tuple_get(TriadTuple *t, int32_t idx);
void        triad_tuple_set(TriadTuple *t, int32_t idx, TriadValue v);
int32_t     triad_tuple_len(TriadTuple *t);
bool        triad_tuple_eq(TriadTuple *a, TriadTuple *b);
int32_t     triad_tuple_hash(TriadTuple *t);

TriadBytes *triad_bytes_new(const uint8_t *data, int32_t len);
void        triad_bytes_free(TriadBytes *b);
int32_t     triad_bytes_len(TriadBytes *b);
bool        triad_bytes_eq(TriadBytes *a, TriadBytes *b);
int32_t     triad_bytes_hash(TriadBytes *b);
TriadString *triad_bytes_repr(TriadBytes *b);

TriadComplex *triad_complex_new(double re, double im);
void          triad_complex_free(TriadComplex *c);
double        triad_complex_abs(TriadComplex *c);
TriadComplex *triad_complex_add(TriadComplex *a, TriadComplex *b);
TriadComplex *triad_complex_sub(TriadComplex *a, TriadComplex *b);
TriadComplex *triad_complex_mul(TriadComplex *a, TriadComplex *b);
TriadComplex *triad_complex_div(TriadComplex *a, TriadComplex *b);
bool          triad_complex_eq(TriadComplex *a, TriadComplex *b);
TriadString  *triad_complex_repr(TriadComplex *c);

TriadSet   *triad_set_new(void);
void        triad_set_free(TriadSet *s);
void        triad_set_add(TriadSet *s, TriadValue v);
bool        triad_set_has(TriadSet *s, TriadValue v);
int32_t     triad_set_len(TriadSet *s);
TriadList  *triad_set_to_list(TriadSet *s);
bool        triad_set_eq(TriadSet *a, TriadSet *b);

TriadGenerator *triad_generator_new(void);
void            triad_generator_free(TriadGenerator *g);
TriadValue      triad_generator_next(TriadGenerator *g);
bool            triad_generator_done(TriadGenerator *g);

TriadClassMeta *triad_class_meta_new(const char *name, TriadClassMeta *parent,
                                     const char **fields, int32_t fields_len);
void            triad_class_meta_free(TriadClassMeta *cm);
TriadClassMeta *triad_class_meta_lookup(const char *name);
void            triad_class_meta_register(TriadClassMeta *cm);
TriadNativeFn   triad_class_meta_resolve_method(TriadClassMeta *cm, const char *method);
TriadClassMeta *triad_class_meta_get_parent(TriadClassMeta *cm);
TriadValue      triad_object_new_typed(const char *type_name, TriadClassMeta *cm);

TriadValue _triad_yield_value(TriadValue v);
TriadValue _triad_await_value(TriadValue v);
TriadValue _triad_super(int32_t nargs, TriadValue *args);

void        triad_retain(TriadValue *v);
void        triad_release(TriadValue *v);

TriadString *triad_str_new(const char *data);
TriadString *triad_str_new_len(const char *data, int32_t len);
TriadString *triad_str_copy(TriadString *s);
void         triad_str_free(TriadString *s);
TriadString *triad_str_concat(TriadString *a, TriadString *b);
bool         triad_str_eq(TriadString *a, TriadString *b);
int32_t      triad_str_hash(TriadString *s);
TriadString *triad_str_slice(TriadString *s, int32_t start, int32_t end, int32_t step);
TriadString *triad_str_upper(TriadString *s);
TriadString *triad_str_lower(TriadString *s);
TriadString *triad_str_strip(TriadString *s);
TriadString *triad_str_replace(TriadString *s, const char *old, const char *rep);
TriadList   *triad_str_split(TriadString *s, const char *sep);
bool         triad_str_starts_with(TriadString *s, const char *prefix);
bool         triad_str_ends_with(TriadString *s, const char *suffix);
bool         triad_str_contains(TriadString *s, const char *sub);
int32_t      triad_str_len(TriadString *s);
TriadString *triad_str_format(const char *fmt, ...);

TriadRegex *triad_regex_compile(const char *pattern, const char *flags);
void        triad_regex_free(TriadRegex *rx);
bool        triad_regex_test(TriadRegex *rx, const char *s);
TriadValue  triad_regex_search(TriadRegex *rx, const char *s);
TriadValue  triad_regex_match(TriadRegex *rx, const char *s);
TriadValue  triad_regex_findall(TriadRegex *rx, const char *s);
const char *triad_regex_pattern(const TriadRegex *rx);
const char *triad_regex_flags(const TriadRegex *rx);

TriadList  *triad_list_new(void);
TriadList  *triad_list_new_cap(int32_t cap);
void        triad_list_free(TriadList *l);
void        triad_list_push(TriadList *l, TriadValue v);
TriadValue  triad_list_get(TriadList *l, int32_t idx);
void        triad_list_set(TriadList *l, int32_t idx, TriadValue v);
TriadList  *triad_list_slice(TriadList *l, int32_t start, int32_t end, int32_t step);
int32_t     triad_list_len(TriadList *l);
void        triad_list_sort(TriadList *l);
bool        triad_list_contains(TriadList *l, TriadValue v);
TriadList  *triad_list_reversed(TriadList *l);
int32_t     triad_list_index_of(TriadList *l, TriadValue v);
void        triad_list_insert(TriadList *l, int32_t idx, TriadValue v);
void        triad_list_remove_at(TriadList *l, int32_t idx);

TriadDict  *triad_dict_new(void);
void        triad_dict_free(TriadDict *d);
TriadValue  triad_dict_get(TriadDict *d, TriadString *key);
void        triad_dict_set(TriadDict *d, TriadString *key, TriadValue val);
bool        triad_dict_has(TriadDict *d, TriadString *key);
void        triad_dict_del(TriadDict *d, TriadString *key);
int32_t     triad_dict_len(TriadDict *d);
TriadList  *triad_dict_keys(TriadDict *d);
TriadList  *triad_dict_values(TriadDict *d);
TriadList  *triad_dict_items(TriadDict *d);

TriadNDArray *triad_ndarray_new(int32_t ndim, int32_t *shape);
TriadNDArray *triad_ndarray_new_data(int32_t ndim, int32_t *shape, double *data);
TriadNDArray *triad_ndarray_zeros(int32_t ndim, int32_t *shape);
TriadNDArray *triad_ndarray_ones(int32_t ndim, int32_t *shape);
void          triad_ndarray_free(TriadNDArray *a);
TriadNDArray *triad_ndarray_reshape(TriadNDArray *a, int32_t ndim, int32_t *shape);
TriadNDArray *triad_ndarray_slice(TriadNDArray *a, int32_t *starts, int32_t *ends, int32_t *steps);
int64_t       triad_ndarray_size(TriadNDArray *a);
double        triad_ndarray_get(TriadNDArray *a, int32_t *indices);
void          triad_ndarray_set(TriadNDArray *a, int32_t *indices, double val);
double        triad_ndarray_sum(TriadNDArray *a);
double        triad_ndarray_mean(TriadNDArray *a);
double        triad_ndarray_max(TriadNDArray *a);
double        triad_ndarray_min(TriadNDArray *a);

TriadClosure *triad_closure_new(TriadNativeFn fn, int32_t ncaptured);
void          triad_closure_free(TriadClosure *c);
TriadValue    triad_closure_call(TriadClosure *c, int32_t nargs, TriadValue *args);

TriadObject  *triad_object_new(const char *type_name);
void          triad_object_free(TriadObject *o);
TriadValue    triad_object_get(TriadObject *o, TriadString *key);
void          triad_object_set(TriadObject *o, TriadString *key, TriadValue val);

void         triad_print(int32_t nargs, TriadValue *args);
TriadValue   triad_input(TriadString *prompt);

double       triad_math_sqrt(double x);
double       triad_math_sin(double x);
double       triad_math_cos(double x);
double       triad_math_tan(double x);
double       triad_math_exp(double x);
double       triad_math_log(double x);
double       triad_math_log10(double x);
double       triad_math_floor(double x);
double       triad_math_ceil(double x);
double       triad_math_abs(double x);
int64_t      triad_math_abs_int(int64_t x);
int64_t      triad_math_clamp(int64_t x, int64_t lo, int64_t hi);
double       triad_math_pow(double base, double exp);
int64_t      triad_math_min(int64_t a, int64_t b);
int64_t      triad_math_max(int64_t a, int64_t b);

void         triad_random_seed(uint64_t seed);
double       triad_random_double(void);
int64_t      triad_random_int(int64_t lo, int64_t hi);
TriadValue   triad_random_choice(TriadList *l);
void         triad_random_shuffle(TriadList *l);
double       triad_random_uniform(double lo, double hi);

TriadList   *triad_range(int64_t start, int64_t stop, int64_t step);

#define TRIAD_MAX_JMP 32

extern jmp_buf  triad_jmp_bufs[TRIAD_MAX_JMP];
extern int32_t  triad_jmp_depth;
extern TriadValue triad_exception;

void triad_throw(TriadValue exc);

#define TRIAD_TRY_BEGIN \
    do { \
        int32_t _saved_depth = triad_jmp_depth; \
        if (triad_jmp_depth >= TRIAD_MAX_JMP) { fprintf(stderr, "exception depth exceeded\n"); exit(1); } \
        int _jmp = setjmp(triad_jmp_bufs[triad_jmp_depth++]); \
        if (_jmp == 0) {

#define TRIAD_CATCH(var) \
        } else { \
            TriadValue var = triad_exception;

#define TRIAD_FINALLY \
        } \
        triad_jmp_depth = _saved_depth + 1; \
        {

#define TRIAD_END \
        } \
        triad_jmp_depth = _saved_depth; \
    } while(0)

bool         triad_is_y(TriadValue v);
TriadString *triad_value_to_string(TriadValue v);
TriadValue   triad_type_name(TriadValue v);
bool         triad_value_eq(TriadValue a, TriadValue b);

#define TRIAD_PI  3.14159265358979323846
#define TRIAD_E   2.71828182845904523536

typedef struct { double re, im; } TriadCplx;

typedef struct {
    int32_t     N;
    double      L, dt, T;
    double      hbar, m;
    double      omega;
    double      Lambda, alpha, sigma, Gamma, f_FDT;
    int32_t     M;
    double     *nu;
    double     *lam;
    int32_t     mode;
    uint64_t    seed;
    const char *V_ext;
    int32_t     D;
    const char *bc;
    double      bc_width;
    int32_t     fdt_couple;

    double      kT;
    double      init_sigma;
    double      init_k0[3];
    int32_t     init_mode;

    int32_t     record_every;
    const double *init_k0_ext;
    int32_t     init_k0_ext_len;
} TriadSolverC;

typedef struct {
    TriadCplx *psi_final;
    double    *y_final;
    double    *density_final;
    double    *x;
    double     dx;
    double    *density_t;
    int32_t    n_records;
} TriadSolverResult;

TriadSolverResult triad_solve_1d(const TriadSolverC *p);
TriadSolverResult triad_solve_from_psi(const TriadSolverC *p, const TriadCplx *psi_init);
TriadSolverResult triad_solve_from_state(const TriadSolverC *p, const TriadCplx *psi_init,
                                         const double *y_init);
void              triad_solver_result_free(TriadSolverResult *r);

typedef struct {
    TriadCplx *psi_final;
    double    *y_final;
    double    *density_final;
    double    *x;
    double     dx;
} TriadSolverResult2D;

TriadSolverResult2D triad_solve_2d(const TriadSolverC *p);
void triad_solver_result_2d_free(TriadSolverResult2D *r);

typedef struct {
    TriadCplx *psi_final;
    double    *y_final;
    double    *density_final;
    double    *x;
    double     dx;
    double    *peak_t;
    double    *participation_t;
    int32_t    n_records;
} TriadSolverResult3D;

TriadSolverResult3D triad_solve_3d(const TriadSolverC *p);
void triad_solver_result_3d_free(TriadSolverResult3D *r);

typedef struct {
    TriadCplx *psi_final;
    double    *y_final;
    double    *density_final;
    double    *x;
    double     dx;
    double    *peak_t;
    double    *participation_t;
    int32_t    n_records;
} TriadSolverResultND;

TriadSolverResultND triad_solve_nd(const TriadSolverC *p);
void triad_solver_result_nd_free(TriadSolverResultND *r);

void triad_fft_fn(int32_t N, const TriadCplx *in, TriadCplx *out);
void triad_ifft_fn(int32_t N, const TriadCplx *in, TriadCplx *out);
void triad_fftfreq(int32_t N, double dx, double *out);

void triad_fft2_fn(int32_t N, const TriadCplx *in, TriadCplx *out);
void triad_ifft2_fn(int32_t N, const TriadCplx *in, TriadCplx *out);

void triad_fft3_fn(int32_t N, const TriadCplx *in, TriadCplx *out);
void triad_ifft3_fn(int32_t N, const TriadCplx *in, TriadCplx *out);

void triad_apply_potential(int64_t n, TriadCplx *psi,
                           const double *V, double dt_over_hbar);
void triad_density_64(int64_t n, const TriadCplx *psi, double *rho);

void triad_matmul(int64_t M, int64_t K, int64_t N,
                  const double *A, const double *B, double *C);

#endif
