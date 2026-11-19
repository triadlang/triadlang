/*
 * triad_check_legacy.h — Native legacy (v1 DSL) typecheck.
 *
 * Mirrors compiler/typecheck.py. Walks a TriadLegacyNode program once,
 * collects all errors with byte-identical text format to TypeCheckError,
 * returns -1 if any error was emitted.
 *
 * The Python checker takes a `regime_registry: set[str]`. Native API
 * accepts the same as an array of cstrings (regime_names) of length
 * regime_count. Pass NULL/0 to disable regime checks (rare).
 *
 * Ownership: error strings are allocated in the supplied arena.
 */
#ifndef TRIAD_CHECK_LEGACY_H
#define TRIAD_CHECK_LEGACY_H

#include "triad_frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char **items;   /* arena-owned strings */
    size_t       len;
} TriadCheckLegacyErrors;

/* Run scope + metric/predicate/projection checks on a parsed legacy
 * Program. Returns 0 on success, -1 if any error was emitted.
 *
 *   regime_names: array of valid regime names (sorted not required).
 *   regime_count: length of regime_names.
 *
 * `out` may be NULL to discard the error list (return value still
 * indicates pass/fail). */
int triad_check_legacy_program(TriadArena              *arena,
                               const TriadLegacyNode   *program,
                               const char *const       *regime_names,
                               size_t                   regime_count,
                               TriadCheckLegacyErrors  *out);

#ifdef __cplusplus
}
#endif

#endif /* TRIAD_CHECK_LEGACY_H */
