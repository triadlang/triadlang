/*
 * triad_check.h — Native universal typecheck (mirror of
 * compiler/typecheck_universal.py).
 *
 * Scope + arity only — not a type system. Reports errors with
 * (file, line, col) and the same E2001/E2002/E2010/E2011 codes used by
 * the Python checker, so the native CLI `triad check` can match
 * `python -m cli.main check` output later.
 *
 * Ownership: all error strings live in the supplied arena.
 */
#ifndef TRIAD_CHECK_H
#define TRIAD_CHECK_H

#include "triad_frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char **items;   /* arena-owned strings */
    size_t       len;
} TriadCheckErrors;

/* Run scope + arity checks on a parsed AST module.
 * Returns 0 if no errors, -1 otherwise. Errors (if any) land in *out.
 * `out` may be NULL to discard the list (return value still indicates
 * pass/fail). */
int triad_check_module(TriadArena         *arena,
                       const TriadAstNode *module,
                       TriadCheckErrors   *out);

#ifdef __cplusplus
}
#endif

#endif /* TRIAD_CHECK_H */
