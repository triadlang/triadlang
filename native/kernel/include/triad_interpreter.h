#ifndef TRIAD_INTERPRETER_H
#define TRIAD_INTERPRETER_H

#include "triad_kernel.h"
#include "triad_tri_loader.h"

#define INTERP_MAX_VARS   256
#define INTERP_MAX_SCOPE  32
#define INTERP_MAX_CALLS  64
#define INTERP_STACK_SIZE 4096

typedef enum {
    VAL_NONE,
    VAL_INT,
    VAL_FLOAT,
    VAL_STRING,
    VAL_BOOL,
    VAL_LIST,
    VAL_MAP,
    VAL_CLOSURE,
    VAL_NATIVE,
} ValType;

typedef struct TriadVal TriadVal;

typedef struct {
    char name[64];
    TriadVal *value;
} TriadVar;

typedef struct {
    TriadVar vars[INTERP_MAX_VARS];
    int n_vars;
    int parent;
} TriadScope;

typedef struct {
    const char *src;
    int pos;
    int line;
    int col;
    int len;
    TriToken current;
    TriToken peek;
} InterpLexer;

typedef struct {
    TriadScope scopes[INTERP_MAX_SCOPE];
    int scope_top;
    int call_stack[INTERP_MAX_CALLS];
    int call_top;
    TriadVal *stack[INTERP_STACK_SIZE];
    int stack_top;
    int proc_id;
    bool running;
    bool error;
    char error_msg[256];
} TriadInterp;

typedef struct TriadVal {
    ValType type;
    union {
        int64_t  ival;
        double   fval;
        char     sval[256];
        bool     bval;
        struct {
            TriadVal **items;
            int count;
        } list;
        struct {
            char    **keys;
            TriadVal **vals;
            int count;
        } map;
        struct {
            char *params[16];
            int n_params;
            const char *body_src;
            int body_pos;
        } closure;
    };
} TriadVal;

void triad_interp_init(TriadInterp *interp, int proc_id);
int  triad_interp_run(TriadInterp *interp, const char *src);
TriadVal *triad_interp_eval(TriadInterp *interp, const char *expr);
void triad_interp_set_var(TriadInterp *interp, const char *name, TriadVal *val);
TriadVal *triad_interp_get_var(TriadInterp *interp, const char *name);
void triad_interp_push_scope(TriadInterp *interp);
void triad_interp_pop_scope(TriadInterp *interp);
void triad_interp_print(TriadVal *val);

#endif