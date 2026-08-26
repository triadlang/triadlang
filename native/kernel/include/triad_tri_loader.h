#ifndef TRIAD_TRI_LOADER_H
#define TRIAD_TRI_LOADER_H

#include "triad_kernel.h"

typedef enum {
    TRI_TOKEN_IDENT,
    TRI_TOKEN_KEYWORD,
    TRI_TOKEN_NUMBER,
    TRI_TOKEN_STRING,
    TRI_TOKEN_SYMBOL,
    TRI_TOKEN_FSTRING,
    TRI_TOKEN_EOF,
} TriTokenType;

typedef struct {
    TriTokenType type;
    char         value[128];
    int          line;
    int          col;
} TriToken;

typedef struct {
    const char *src;
    int         pos;
    int         line;
    int         col;
    int         len;
    TriToken    current;
} TriLexer;

typedef struct {
    char name[64];
    char source_path[256];
    int  n_functions;
    int  n_imports;
    int  n_statements;
    bool has_solver;
    bool has_observe;
    bool has_reg;
    bool has_couple;
    int  n_capabilities_declared;
    char caps_str[256];
} TriModuleInfo;

void triad_tri_lex_init(TriLexer *lex, const char *src);
int  triad_tri_lex_next(TriLexer *lex, TriToken *out);
int  triad_tri_parse(const char *src, const char *path, TriModuleInfo *out);
int  triad_tri_load_from_vfs(const char *vfs_path, TriModuleInfo *out);

#endif