#ifndef TRIAD_LSP_H
#define TRIAD_LSP_H

#include "triad_frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *name;
    const char *kind;
    int         line;
    int         col;
} TriadLspSymbol;

void triad_lsp_parse_symbols(TriadArena *arena, const char *text,
                             TriadLspSymbol **out, size_t *out_len);

const char *triad_lsp_word_at(TriadArena *arena, const char *line,
                              int char_pos);

const char *triad_lsp_completions_json(TriadArena *arena, const char *text,
                                       int line_num, int char_pos);

const char *triad_lsp_definition_json(TriadArena *arena, const char *text,
                                      int line_num, int char_pos,
                                      const char *uri);

const char *triad_lsp_hover_json(TriadArena *arena, const char *text,
                                 int line_num, int char_pos);

const char *triad_lsp_document_symbols_json(TriadArena *arena,
                                            const char *text,
                                            const char *uri);

const char *triad_lsp_initialize_result_json(TriadArena *arena);

#ifdef __cplusplus
}
#endif

#endif
