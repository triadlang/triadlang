/*
 * triad_lsp.h — Native port of cli/lsp.py functional core.
 *
 * Mirrors the pure functions of TriadLSP:
 *   - parse_symbols(text) → list of {name,kind,line,col}
 *   - word_at(line, char) → string or NULL
 *   - completions(text, line, char) → JSON string
 *   - definition(text, line, char, uri) → JSON string or NULL
 *   - hover(text, line, char) → JSON string or NULL
 *   - document_symbols(text, uri) → JSON string
 *   - initialize_result() → JSON string
 *
 * The stdio JSON-RPC loop is a CLI wrapper on top of these.
 *
 * Output is byte-identical to the Python implementation on the same
 * inputs (same key order in JSON, same set ordering).
 *
 * Ownership: all returned cstrings are allocated in the supplied arena.
 */
#ifndef TRIAD_LSP_H
#define TRIAD_LSP_H

#include "triad_frontend.h"   /* TriadArena */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *name;
    const char *kind;   /* "variable" | "function" | "class" */
    int         line;
    int         col;
} TriadLspSymbol;

/* Extract top-level symbols from text. Returns arena-owned array. */
void triad_lsp_parse_symbols(TriadArena *arena, const char *text,
                             TriadLspSymbol **out, size_t *out_len);

/* Return the identifier-like word touching column `char_pos` on
 * `line`, or NULL if there's no word there. Result is arena-owned. */
const char *triad_lsp_word_at(TriadArena *arena, const char *line,
                              int char_pos);

/* Return the "result" payload of textDocument/completion as a JSON
 * cstring (the object {"isIncomplete":false,"items":[...]}). */
const char *triad_lsp_completions_json(TriadArena *arena, const char *text,
                                       int line_num, int char_pos);

/* Return the "result" payload of textDocument/definition or NULL if
 * no definition was found. */
const char *triad_lsp_definition_json(TriadArena *arena, const char *text,
                                      int line_num, int char_pos,
                                      const char *uri);

/* Return the "result" payload of textDocument/hover or NULL. */
const char *triad_lsp_hover_json(TriadArena *arena, const char *text,
                                 int line_num, int char_pos);

/* Return the "result" payload of textDocument/documentSymbol. */
const char *triad_lsp_document_symbols_json(TriadArena *arena,
                                            const char *text,
                                            const char *uri);

/* Return the "result" payload of initialize (capabilities object). */
const char *triad_lsp_initialize_result_json(TriadArena *arena);

#ifdef __cplusplus
}
#endif

#endif /* TRIAD_LSP_H */
