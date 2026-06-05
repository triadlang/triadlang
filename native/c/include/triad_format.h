/*
 * triad_format.h — Native formatter for the universal TriadLang AST.
 *
 * Mirrors compiler/formatter.py:format_universal. Re-emits the AST as
 * canonically-indented source code. Output is byte-identical to the
 * Python formatter over all fixtures shipped with the repo.
 *
 * Two helpers are exposed as part of the public API because they are
 * the most pedantic parts of byte-equality and are reused by tests:
 *
 *   triad_py_repr_float(v, out, outlen) — mirror Python repr(float)
 *   triad_py_repr_string(s, len, arena) — mirror Python repr(str)
 *
 * Ownership:
 *   - triad_format_module returns a NUL-terminated string allocated in
 *     the supplied arena. Free the arena to free the result.
 *   - triad_py_repr_string returns an arena-owned string as well.
 */
#ifndef TRIAD_FORMAT_H
#define TRIAD_FORMAT_H

#include "triad_frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Format a parsed universal AST module. Returns NULL on failure (e.g.
 * NULL input). Output is "\n"-joined statements with a trailing "\n",
 * matching format_universal's contract. */
const char *triad_format_module(TriadArena         *arena,
                                const TriadAstNode *module);

/* Python-style repr(float). Writes at most outlen-1 chars + NUL.
 * Conventions:
 *   - 0.0 / -0.0           -> "0.0" / "-0.0"
 *   - NaN / +-inf          -> "nan" / "inf" / "-inf"
 *   - shortest decimal that roundtrips, in fixed notation if
 *     |v| in [1e-4, 1e16), scientific otherwise with "e+NN" / "e-NN"
 *     and a 2-digit minimum exponent.
 *   - if the shortest form has no '.' and no exponent, ".0" is appended.
 */
void triad_py_repr_float(double v, char *out, size_t outlen);

/* Python-style repr(str). Chooses single or double quotes by mimicking
 * CPython's PyUnicode_Repr rules, escapes \\ \n \t \r and the quote
 * character. Returns an arena-owned NUL-terminated cstring (with
 * surrounding quotes included). */
const char *triad_py_repr_string(TriadArena *arena,
                                 const char *s,
                                 size_t      len);

#ifdef __cplusplus
}
#endif

#endif /* TRIAD_FORMAT_H */
