#ifndef TRIAD_KERNEL_RUNTIME_H
#define TRIAD_KERNEL_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __KERNEL__
#define TRIAD_NO_BOEHM 1
#define TRIAD_NO_FILEIO 1
#define TRIAD_NO_PYTHON 1
#endif

#include "triad_rt.h"
#include "triad_frontend.h"
#include "triad_compiler.h"

typedef struct {
    TriadArena arena;
    TriadAstNode *ast;
    char *source;
    size_t source_len;
    bool compiled;
} TriadKernelModule;

void triad_kernel_init(void);

void *triad_kernel_alloc(size_t size);
void triad_kernel_free(void *ptr);

TriadKernelModule *triad_kernel_parse(const char *source, size_t len);
TriadValue triad_kernel_eval(TriadKernelModule *mod);
void triad_kernel_module_free(TriadKernelModule *mod);

int triad_kernel_run_file(const char *path);

TriadValue triad_kernel_call(TriadKernelModule *mod, const char *fn_name,
                             int nargs, TriadValue *args);

void triad_kernel_print_value(TriadValue val);

int triad_kernel_register_builtin(const char *name, TriadNativeFn fn);

void triad_kernel_set_print_fn(void (*fn)(const char *msg));
void triad_kernel_set_input_fn(char *(*fn)(char *buf, size_t len));

#endif