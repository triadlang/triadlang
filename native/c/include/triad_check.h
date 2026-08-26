#ifndef TRIAD_CHECK_H
#define TRIAD_CHECK_H

#include "triad_frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char **items;
    size_t       len;
} TriadCheckErrors;

int triad_check_module(TriadArena         *arena,
                       const TriadAstNode *module,
                       TriadCheckErrors   *out);

#ifdef __cplusplus
}
#endif

#endif
