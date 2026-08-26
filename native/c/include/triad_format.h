#ifndef TRIAD_FORMAT_H
#define TRIAD_FORMAT_H

#include "triad_frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

const char *triad_format_module(TriadArena         *arena,
                                const TriadAstNode *module);

void triad_py_repr_float(double v, char *out, size_t outlen);

const char *triad_py_repr_string(TriadArena *arena,
                                 const char *s,
                                 size_t      len);

#ifdef __cplusplus
}
#endif

#endif
