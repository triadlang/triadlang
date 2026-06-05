/* test_lexer.c — Native lexer self-tests. */
#include "triad_frontend.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static TriadTokenList tokenize_or_die(TriadArena *a, const char *src) {
    TriadTokenList toks = {0};
    TriadDiag diag = {0};
    int rc = triad_tokenize(a, src, "<test>", &toks, &diag);
    if (rc != 0) {
        fprintf(stderr, "lex error: %s @ L%d:C%d\n",
                diag.msg ? diag.msg : "?", diag.line, diag.col);
        exit(1);
    }
    return toks;
}

static void test_basic_tokens(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadTokenList t = tokenize_or_die(&a, "let x = 42");
    CHECK(t.len == 5); /* let x = 42 EOF */
    CHECK(t.items[0].kind == TRIAD_TOK_KEYWORD && strcmp(t.items[0].text, "let") == 0);
    CHECK(t.items[1].kind == TRIAD_TOK_IDENT   && strcmp(t.items[1].text, "x") == 0);
    CHECK(t.items[2].kind == TRIAD_TOK_SYMBOL  && strcmp(t.items[2].text, "=") == 0);
    CHECK(t.items[3].kind == TRIAD_TOK_NUMBER  && strcmp(t.items[3].text, "42") == 0);
    CHECK(t.items[4].kind == TRIAD_TOK_EOF);
    triad_arena_free(&a);
}

static void test_keywords_vs_idents(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadTokenList t = tokenize_or_die(&a, "reg entity world couple ring observe run xyz");
    for (size_t k = 0; k < 7; ++k) CHECK(t.items[k].kind == TRIAD_TOK_KEYWORD);
    CHECK(t.items[7].kind == TRIAD_TOK_IDENT);
    triad_arena_free(&a);
}

static void test_string_and_escapes(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadTokenList t = tokenize_or_die(&a, "\"a\\nb\\t\\\"c\"");
    CHECK(t.len == 2);
    CHECK(t.items[0].kind == TRIAD_TOK_STRING);
    CHECK(strcmp(t.items[0].text, "a\nb\t\"c") == 0);
    triad_arena_free(&a);
}

static void test_fstring_parts(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadTokenList t = tokenize_or_die(&a, "f\"x={n+1} done\"");
    CHECK(t.len == 2);
    CHECK(t.items[0].kind == TRIAD_TOK_FSTRING);
    CHECK(t.items[0].parts_len == 3);
    CHECK(t.items[0].parts[0].is_expr == 0 && strcmp(t.items[0].parts[0].text, "x=") == 0);
    CHECK(t.items[0].parts[1].is_expr == 1 && strcmp(t.items[0].parts[1].text, "n+1") == 0);
    CHECK(t.items[0].parts[2].is_expr == 0 && strcmp(t.items[0].parts[2].text, " done") == 0);
    triad_arena_free(&a);
}

static void test_numbers(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadTokenList t = tokenize_or_die(&a, "42 3.14 1e-9 0x2A 0b1010");
    CHECK(t.len == 6);
    CHECK(strcmp(t.items[0].text, "42")    == 0);
    CHECK(strcmp(t.items[1].text, "3.14")  == 0);
    CHECK(strcmp(t.items[2].text, "1e-9")  == 0);
    CHECK(strcmp(t.items[3].text, "0x2A")  == 0);
    CHECK(strcmp(t.items[4].text, "0b1010")== 0);
    triad_arena_free(&a);
}

static void test_negative_number_after_op(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    /* After "=" the "-1" should fuse into a single NUMBER. */
    TriadTokenList t = tokenize_or_die(&a, "x = -1");
    CHECK(t.len == 4); /* x = -1 EOF */
    CHECK(t.items[2].kind == TRIAD_TOK_NUMBER && strcmp(t.items[2].text, "-1") == 0);

    /* "a-1" must remain a 3-token subtraction. */
    TriadTokenList t2 = tokenize_or_die(&a, "a-1");
    CHECK(t2.len == 4);
    CHECK(t2.items[0].kind == TRIAD_TOK_IDENT);
    CHECK(t2.items[1].kind == TRIAD_TOK_SYMBOL && strcmp(t2.items[1].text, "-") == 0);
    CHECK(t2.items[2].kind == TRIAD_TOK_NUMBER && strcmp(t2.items[2].text, "1") == 0);
    triad_arena_free(&a);
}

static void test_multi_char_symbols(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadTokenList t = tokenize_or_die(&a, "== != <= >= -> ** +=");
    const char *expected[] = { "==", "!=", "<=", ">=", "->", "**", "+=" };
    for (size_t k = 0; k < 7; ++k) {
        CHECK(t.items[k].kind == TRIAD_TOK_SYMBOL);
        CHECK(strcmp(t.items[k].text, expected[k]) == 0);
    }
    triad_arena_free(&a);
}

static void test_position_tracking(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadTokenList t = tokenize_or_die(&a, "let x\n  = 1");
    /* let@(1,1) x@(1,5) =@(2,3) 1@(2,5) */
    CHECK(t.items[0].line == 1 && t.items[0].col == 1);
    CHECK(t.items[1].line == 1 && t.items[1].col == 5);
    CHECK(t.items[2].line == 2 && t.items[2].col == 3);
    CHECK(t.items[3].line == 2 && t.items[3].col == 5);
    triad_arena_free(&a);
}

static void test_comments(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadTokenList t = tokenize_or_die(&a,
        "// line comment\n"
        "/* block\n   comment */\n"
        "# python-style\n"
        "x");
    CHECK(t.len == 2);
    CHECK(t.items[0].kind == TRIAD_TOK_IDENT && strcmp(t.items[0].text, "x") == 0);
    triad_arena_free(&a);
}

static void test_lex_error_unterminated_string(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadTokenList toks = {0};
    TriadDiag diag = {0};
    int rc = triad_tokenize(&a, "\"abc", "<t>", &toks, &diag);
    CHECK(rc != 0);
    CHECK(diag.msg && strstr(diag.msg, "unterminated") != NULL);
    triad_arena_free(&a);
}

int main(void) {
    test_basic_tokens();
    test_keywords_vs_idents();
    test_string_and_escapes();
    test_fstring_parts();
    test_numbers();
    test_negative_number_after_op();
    test_multi_char_symbols();
    test_position_tracking();
    test_comments();
    test_lex_error_unterminated_string();
    if (failures == 0) {
        printf("test_lexer: OK\n");
        return 0;
    }
    fprintf(stderr, "test_lexer: %d FAILURE(S)\n", failures);
    return 1;
}
