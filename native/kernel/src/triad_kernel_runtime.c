#include "triad_kernel_runtime.h"
#include "triad_lexer.c"
#include "triad_parser.c"
#include "triad_ast.c"
#include "triad_string.c"
#include "triad_list.c"
#include "triad_dict.c"
#include "triad_object.c"
#include "triad_arena.c"
#include "triad_stdlib.c"

#define MAX_MODULES 32
#define MAX_BUILTINS 256

static TriadKernelModule g_modules[MAX_MODULES];
static int g_module_count = 0;

static struct {
    char name[64];
    TriadNativeFn fn;
} g_builtins[MAX_BUILTINS];
static int g_builtin_count = 0;

static void (*g_print_fn)(const char *msg) = NULL;
static char *(*g_input_fn)(char *buf, size_t len) = NULL;

static char g_print_buffer[4096];

static void *kernel_alloc(size_t size) {
    extern void *heap_alloc(size_t size);
    return heap_alloc(size);
}

static void kernel_free(void *ptr) {
    extern void heap_free(void *ptr);
    heap_free(ptr);
}

static int kernel_print_str(const char *s) {
    if (g_print_fn) {
        g_print_fn(s);
        return (int)strlen(s);
    }
    extern void vga_puts(const char *s);
    vga_puts(s);
    return (int)strlen(s);
}

static int kernel_print_int(int64_t v) {
    char buf[32];
    int i = 30;
    int neg = 0;
    if (v < 0) { neg = 1; v = -v; }
    if (v == 0) { buf[i--] = '0'; }
    else { while (v > 0) { buf[i--] = '0' + (v % 10); v /= 10; } }
    if (neg) buf[i--] = '-';
    buf[31] = '\0';
    kernel_print_str(&buf[i+1]);
    return 30 - i;
}

static int kernel_print_float(double v) {
    char buf[64];
    int len = snprintf(buf, sizeof(buf), "%g", v);
    kernel_print_str(buf);
    return len;
}

static TriadValue builtin_print(int nargs, TriadValue *args) {
    for (int i = 0; i < nargs; i++) {
        if (i > 0) kernel_print_str(" ");
        TriadString *s = triad_value_to_string(args[i]);
        if (s) {
            kernel_print_str(s->data);
            triad_str_free(s);
        }
    }
    kernel_print_str("\n");
    return TRIAD_NONE_VAL;
}

static TriadValue builtin_input(int nargs, TriadValue *args) {
    if (nargs > 0 && args[0].tag == TRIAD_STRING && args[0].as.sval) {
        kernel_print_str(args[0].as.sval->data);
    }
    char buf[4096];
    if (g_input_fn) {
        g_input_fn(buf, sizeof(buf));
    } else {
        int i = 0;
        extern int keyboard_get_key(void);
        while (i < (int)sizeof(buf) - 1) {
            int c = keyboard_get_key();
            if (c == '\n' || c == '\r') break;
            if (c >= 32 && c < 127) buf[i++] = (char)c;
        }
        buf[i] = '\0';
    }
    TriadString *s = triad_str_new_len(buf, (int32_t)strlen(buf));
    return (TriadValue){.tag = TRIAD_STRING, .as = {.sval = s}};
}

static TriadValue builtin_str(int nargs, TriadValue *args) {
    if (nargs != 1) return TRIAD_NONE_VAL;
    TriadString *s = triad_value_to_string(args[0]);
    return (TriadValue){.tag = TRIAD_STRING, .as = {.sval = s}};
}

static TriadValue builtin_int(int nargs, TriadValue *args) {
    if (nargs != 1) return TRIAD_NONE_VAL;
    switch (args[0].tag) {
        case TRIAD_INT: return args[0];
        case TRIAD_FLOAT: return TRIAD_INT((int64_t)args[0].as.fval);
        case TRIAD_STRING: {
            int64_t v = 0;
            if (args[0].as.sval && args[0].as.sval->data) {
                v = (int64_t)strtoll(args[0].as.sval->data, NULL, 10);
            }
            return TRIAD_INT(v);
        }
        default: return TRIAD_INT(0);
    }
}

static TriadValue builtin_float(int nargs, TriadValue *args) {
    if (nargs != 1) return TRIAD_NONE_VAL;
    switch (args[0].tag) {
        case TRIAD_FLOAT: return args[0];
        case TRIAD_INT: return TRIAD_FLOAT((double)args[0].as.ival);
        case TRIAD_STRING: {
            double v = 0.0;
            if (args[0].as.sval && args[0].as.sval->data) {
                v = strtod(args[0].as.sval->data, NULL);
            }
            return TRIAD_FLOAT(v);
        }
        default: return TRIAD_FLOAT(0.0);
    }
}

static TriadValue builtin_len(int nargs, TriadValue *args) {
    if (nargs != 1) return TRIAD_INT(0);
    switch (args[0].tag) {
        case TRIAD_STRING: return TRIAD_INT(args[0].as.sval ? args[0].as.sval->len : 0);
        case TRIAD_LIST: return TRIAD_INT(args[0].as.lval ? args[0].as.lval->len : 0);
        case TRIAD_DICT: return TRIAD_INT(args[0].as.dval ? args[0].as.dval->len : 0);
        default: return TRIAD_INT(0);
    }
}

static TriadValue builtin_range(int nargs, TriadValue *args) {
    int64_t start = 0, stop = 0, step = 1;
    if (nargs == 1) {
        if (args[0].tag != TRIAD_INT) return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = triad_list_new()}};
        stop = args[0].as.ival;
    } else if (nargs == 2) {
        if (args[0].tag != TRIAD_INT || args[1].tag != TRIAD_INT)
            return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = triad_list_new()}};
        start = args[0].as.ival;
        stop = args[1].as.ival;
    } else if (nargs >= 3) {
        if (args[0].tag != TRIAD_INT || args[1].tag != TRIAD_INT || args[2].tag != TRIAD_INT)
            return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = triad_list_new()}};
        start = args[0].as.ival;
        stop = args[1].as.ival;
        step = args[2].as.ival;
    }
    TriadList *l = triad_range(start, stop, step);
    return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = l}};
}

static TriadValue builtin_type(int nargs, TriadValue *args) {
    if (nargs != 1) return TRIAD_NONE_VAL;
    return triad_type_name(args[0]);
}

void triad_kernel_init(void) {
    triad_runtime_init();
    g_module_count = 0;
    g_builtin_count = 0;
    
    triad_kernel_register_builtin("print", builtin_print);
    triad_kernel_register_builtin("input", builtin_input);
    triad_kernel_register_builtin("str", builtin_str);
    triad_kernel_register_builtin("int", builtin_int);
    triad_kernel_register_builtin("float", builtin_float);
    triad_kernel_register_builtin("len", builtin_len);
    triad_kernel_register_builtin("range", builtin_range);
    triad_kernel_register_builtin("type", builtin_type);
}

void *triad_kernel_alloc(size_t size) {
    return kernel_alloc(size);
}

void triad_kernel_free(void *ptr) {
    kernel_free(ptr);
}

TriadKernelModule *triad_kernel_parse(const char *source, size_t len) {
    if (g_module_count >= MAX_MODULES) return NULL;
    
    TriadKernelModule *mod = &g_modules[g_module_count];
    triad_arena_init(&mod->arena, 65536);
    
    mod->source_len = len;
    mod->source = (char *)kernel_alloc(len + 1);
    if (!mod->source) return NULL;
    memcpy(mod->source, source, len);
    mod->source[len] = '\0';
    
    TriadDiag diag;
    memset(&diag, 0, sizeof(diag));
    
    mod->ast = triad_parse_source(&mod->arena, source, "<kernel>", &diag);
    if (!mod->ast) {
        kernel_free(mod->source);
        triad_arena_free(&mod->arena);
        return NULL;
    }
    
    mod->compiled = false;
    g_module_count++;
    return mod;
}

static TriadValue eval_node(TriadAstNode *node, TriadArena *arena);

static TriadValue eval_int_lit(TriadAstNode *node) {
    return TRIAD_INT(node->as.int_val);
}

static TriadValue eval_float_lit(TriadAstNode *node) {
    return TRIAD_FLOAT(node->as.float_val);
}

static TriadValue eval_string_lit(TriadAstNode *node) {
    TriadString *s = triad_str_new(node->as.string_val);
    return (TriadValue){.tag = TRIAD_STRING, .as = {.sval = s}};
}

static TriadValue eval_bool_lit(TriadAstNode *node) {
    return TRIAD_BOOL(node->as.bool_val);
}

static TriadValue eval_none_lit(void) {
    return TRIAD_NONE_VAL;
}

static TriadValue eval_list(TriadAstNode *node, TriadArena *arena) {
    TriadList *l = triad_list_new();
    for (size_t i = 0; i < node->as.list.elements_len; i++) {
        TriadValue v = eval_node(node->as.list.elements[i], arena);
        triad_list_push(l, v);
    }
    return (TriadValue){.tag = TRIAD_LIST, .as = {.lval = l}};
}

static TriadValue eval_binop(TriadAstNode *node, TriadArena *arena) {
    TriadValue left = eval_node(node->as.op.left, arena);
    TriadValue right = eval_node(node->as.op.right, arena);
    
    const char *op = node->as.op.op;
    
    if (left.tag == TRIAD_INT && right.tag == TRIAD_INT) {
        int64_t a = left.as.ival, b = right.as.ival;
        if (strcmp(op, "+") == 0) return TRIAD_INT(a + b);
        if (strcmp(op, "-") == 0) return TRIAD_INT(a - b);
        if (strcmp(op, "*") == 0) return TRIAD_INT(a * b);
        if (strcmp(op, "/") == 0) return TRIAD_INT(b != 0 ? a / b : 0);
        if (strcmp(op, "%") == 0) return TRIAD_INT(b != 0 ? a % b : 0);
        if (strcmp(op, "<") == 0) return TRIAD_BOOL(a < b);
        if (strcmp(op, ">") == 0) return TRIAD_BOOL(a > b);
        if (strcmp(op, "<=") == 0) return TRIAD_BOOL(a <= b);
        if (strcmp(op, ">=") == 0) return TRIAD_BOOL(a >= b);
        if (strcmp(op, "==") == 0) return TRIAD_BOOL(a == b);
        if (strcmp(op, "!=") == 0) return TRIAD_BOOL(a != b);
        if (strcmp(op, "and") == 0) return TRIAD_BOOL(a && b);
        if (strcmp(op, "or") == 0) return TRIAD_BOOL(a || b);
    }
    
    if (left.tag == TRIAD_FLOAT || right.tag == TRIAD_FLOAT) {
        double a = (left.tag == TRIAD_FLOAT) ? left.as.fval : (double)left.as.ival;
        double b = (right.tag == TRIAD_FLOAT) ? right.as.fval : (double)right.as.ival;
        if (strcmp(op, "+") == 0) return TRIAD_FLOAT(a + b);
        if (strcmp(op, "-") == 0) return TRIAD_FLOAT(a - b);
        if (strcmp(op, "*") == 0) return TRIAD_FLOAT(a * b);
        if (strcmp(op, "/") == 0) return TRIAD_FLOAT(b != 0 ? a / b : 0);
        if (strcmp(op, "<") == 0) return TRIAD_BOOL(a < b);
        if (strcmp(op, ">") == 0) return TRIAD_BOOL(a > b);
        if (strcmp(op, "<=") == 0) return TRIAD_BOOL(a <= b);
        if (strcmp(op, ">=") == 0) return TRIAD_BOOL(a >= b);
        if (strcmp(op, "==") == 0) return TRIAD_BOOL(a == b);
        if (strcmp(op, "!=") == 0) return TRIAD_BOOL(a != b);
    }
    
    if (left.tag == TRIAD_STRING && right.tag == TRIAD_STRING) {
        if (strcmp(op, "+") == 0) {
            TriadString *s = triad_str_concat(left.as.sval, right.as.sval);
            return (TriadValue){.tag = TRIAD_STRING, .as = {.sval = s}};
        }
        if (strcmp(op, "==") == 0) return TRIAD_BOOL(triad_str_eq(left.as.sval, right.as.sval));
        if (strcmp(op, "!=") == 0) return TRIAD_BOOL(!triad_str_eq(left.as.sval, right.as.sval));
    }
    
    return TRIAD_NONE_VAL;
}

static TriadValue eval_call(TriadAstNode *node, TriadArena *arena) {
    TriadValue fn_val = eval_node(node->as.call.func_or_obj, arena);
    
    int nargs = (int)node->as.call.args_len;
    TriadValue *args = NULL;
    
    if (nargs > 0) {
        args = (TriadValue *)kernel_alloc(nargs * sizeof(TriadValue));
        for (int i = 0; i < nargs; i++) {
            args[i] = eval_node(node->as.call.args[i], arena);
        }
    }
    
    TriadValue result = TRIAD_NONE_VAL;
    
    if (fn_val.tag == TRIAD_NATIVE_FN) {
        result = fn_val.as.nfn(nargs, args);
    } else if (fn_val.tag == TRIAD_CLOSURE) {
        result = triad_closure_call(fn_val.as.cval, nargs, args);
    }
    
    if (args) kernel_free(args);
    return result;
}

static TriadValue eval_ident(TriadAstNode *node) {
    for (int i = g_builtin_count - 1; i >= 0; i--) {
        if (strcmp(g_builtins[i].name, node->as.ident_name) == 0) {
            return (TriadValue){.tag = TRIAD_NATIVE_FN, .as = {.nfn = g_builtins[i].fn}};
        }
    }
    return TRIAD_NONE_VAL;
}

static TriadValue eval_node(TriadAstNode *node, TriadArena *arena) {
    if (!node) return TRIAD_NONE_VAL;
    
    switch (node->kind) {
        case TRIAD_AST_INT_LIT: return eval_int_lit(node);
        case TRIAD_AST_FLOAT_LIT: return eval_float_lit(node);
        case TRIAD_AST_STRING_LIT: return eval_string_lit(node);
        case TRIAD_AST_BOOL_LIT: return eval_bool_lit(node);
        case TRIAD_AST_NONE_LIT: return eval_none_lit();
        case TRIAD_AST_LIST: return eval_list(node, arena);
        case TRIAD_AST_BINOP: return eval_binop(node, arena);
        case TRIAD_AST_CALL: return eval_call(node, arena);
        case TRIAD_AST_IDENT: return eval_ident(node);
        default: return TRIAD_NONE_VAL;
    }
}

TriadValue triad_kernel_eval(TriadKernelModule *mod) {
    if (!mod || !mod->ast) return TRIAD_NONE_VAL;
    
    if (mod->ast->kind == TRIAD_AST_MODULE) {
        TriadValue last = TRIAD_NONE_VAL;
        for (size_t i = 0; i < mod->ast->as.module.body_len; i++) {
            last = eval_node(mod->ast->as.module.body[i], &mod->arena);
        }
        return last;
    }
    
    return eval_node(mod->ast, &mod->arena);
}

void triad_kernel_module_free(TriadKernelModule *mod) {
    if (!mod) return;
    if (mod->source) kernel_free(mod->source);
    triad_arena_free(&mod->arena);
    mod->ast = NULL;
    mod->source = NULL;
}

int triad_kernel_run_file(const char *path) {
    extern int disk_read(uint64_t sector, void *buf, size_t count);
    
    char *buf = (char *)kernel_alloc(65536);
    if (!buf) return -1;
    
    size_t len = 0;
    // Simplified file reading - in real kernel would use VFS
    // For now, just read from embedded source
    
    TriadKernelModule *mod = triad_kernel_parse(buf, len);
    kernel_free(buf);
    
    if (!mod) return -2;
    
    triad_kernel_eval(mod);
    triad_kernel_module_free(mod);
    return 0;
}

TriadValue triad_kernel_call(TriadKernelModule *mod, const char *fn_name,
                            int nargs, TriadValue *args) {
    (void)mod;
    (void)fn_name;
    (void)nargs;
    (void)args;
    return TRIAD_NONE_VAL;
}

void triad_kernel_print_value(TriadValue val) {
    TriadString *s = triad_value_to_string(val);
    if (s) {
        kernel_print_str(s->data);
        triad_str_free(s);
    }
}

int triad_kernel_register_builtin(const char *name, TriadNativeFn fn) {
    if (g_builtin_count >= MAX_BUILTINS) return -1;
    snprintf(g_builtins[g_builtin_count].name, sizeof(g_builtins[0].name), "%s", name);
    g_builtins[g_builtin_count].fn = fn;
    g_builtin_count++;
    return 0;
}

void triad_kernel_set_print_fn(void (*fn)(const char *msg)) {
    g_print_fn = fn;
}

void triad_kernel_set_input_fn(char *(*fn)(char *buf, size_t len)) {
    g_input_fn = fn;
}