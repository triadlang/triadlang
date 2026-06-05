/*
 * triad_docgen.h — Native port of compiler/docgen.py.
 *
 * Extracts module/function/class/type/const/import documentation
 * from a .tri source file (string-scanning, mirrors the Python regex
 * model rather than the AST), and renders Markdown or HTML.
 *
 * Output is byte-identical to the Python docgen over every fixture in
 * the repo (validated by native/c/parity_docgen).
 *
 * Ownership:
 *   - triad_docgen_markdown / triad_docgen_html return NUL-terminated
 *     cstrings allocated in the supplied arena.
 */
#ifndef TRIAD_DOCGEN_H
#define TRIAD_DOCGEN_H

#include "triad_frontend.h"   /* TriadArena */

#ifdef __cplusplus
extern "C" {
#endif

/* Render the .tri source as Markdown documentation, identical to
 * compiler.docgen.to_markdown(parse_tri_docs(source), filename).
 * Returns NULL on allocation failure. */
const char *triad_docgen_markdown(TriadArena *arena,
                                  const char *source,
                                  const char *filename);

/* Render the .tri source as HTML documentation, identical to
 * compiler.docgen.to_html(...). */
const char *triad_docgen_html(TriadArena *arena,
                              const char *source,
                              const char *filename);

#ifdef __cplusplus
}
#endif

#endif /* TRIAD_DOCGEN_H */
