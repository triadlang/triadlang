/* test_legacy_frontend.c — Native legacy lexer + parser self-tests. */
#include "triad_frontend.h"

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

static TriadLegacyTokenList tok_or_die(TriadArena *a, const char *src) {
    TriadLegacyTokenList t = {0};
    TriadDiag d = {0};
    if (triad_legacy_tokenize(a, src, &t, &d) != 0) {
        fprintf(stderr, "lex: %s\n", d.msg ? d.msg : "?");
        exit(1);
    }
    return t;
}

static TriadLegacyNode *parse_or_die(TriadArena *a, const char *src) {
    TriadDiag d = {0};
    TriadLegacyNode *p = triad_legacy_parse_source(a, src, &d);
    if (!p) {
        fprintf(stderr, "parse: %s @ L%d:C%d\n",
                d.msg ? d.msg : "?", d.line, d.col);
        exit(1);
    }
    return p;
}

static void test_lex_basic(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadLegacyTokenList t = tok_or_die(&a, "reg n = 5; MOV n, 1; HALT;");
    CHECK(t.items[0].kind == TRIAD_LTOK_KEYWORD && strcmp(t.items[0].text, "reg") == 0);
    CHECK(t.items[1].kind == TRIAD_LTOK_IDENT);
    CHECK(t.items[3].kind == TRIAD_LTOK_NUMBER && strcmp(t.items[3].text, "5") == 0);
    CHECK(t.items[5].kind == TRIAD_LTOK_OPCODE);
    triad_arena_free(&a);
}

static void test_lex_annot(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadLegacyTokenList t = tok_or_die(&a, "@T(10.0) reg n = 5;");
    CHECK(t.items[0].kind == TRIAD_LTOK_ANNOT);
    CHECK(strcmp(t.items[0].text, "T(10.0)") == 0);
    triad_arena_free(&a);
}

static void test_lex_hex_binary_neg(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadLegacyTokenList t = tok_or_die(&a, "0x2A 0b1010 -7");
    CHECK(t.items[0].kind == TRIAD_LTOK_NUMBER && strcmp(t.items[0].text, "42") == 0);
    CHECK(t.items[1].kind == TRIAD_LTOK_NUMBER && strcmp(t.items[1].text, "10") == 0);
    CHECK(t.items[2].kind == TRIAD_LTOK_NUMBER && strcmp(t.items[2].text, "-7") == 0);
    triad_arena_free(&a);
}

static void test_parse_reg_with_regime(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadLegacyNode *p = parse_or_die(&a,
        "reg s: anti_collapse { L: 64.0; N: 256; } = 1;");
    CHECK(p->kind == TRIAD_LAST_PROGRAM);
    CHECK(p->u.program.body_len == 1);
    TriadLegacyNode *r = p->u.program.body[0];
    CHECK(r->kind == TRIAD_LAST_REG_DECL);
    CHECK(strcmp(r->u.reg_decl.name, "s") == 0);
    CHECK(strcmp(r->u.reg_decl.regime_name, "anti_collapse") == 0);
    CHECK(r->u.reg_decl.has_overrides);
    CHECK(r->u.reg_decl.overrides_len == 2);
    CHECK(strcmp(r->u.reg_decl.overrides[0].key, "L") == 0);
    CHECK(r->u.reg_decl.overrides[0].value.kind == TRIAD_LVAL_FLOAT);
    CHECK(r->u.reg_decl.overrides[1].value.kind == TRIAD_LVAL_INT);
    CHECK(r->u.reg_decl.initial != NULL);
    triad_arena_free(&a);
}

static void test_parse_loop_if_else(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadLegacyNode *p = parse_or_die(&a,
        "reg n = 5;\n"
        "loop n { if cond { MUL t, n, n; } else { HALT; } }\n");
    CHECK(p->u.program.body_len == 2);
    CHECK(p->u.program.body[1]->kind == TRIAD_LAST_LOOP_BLOCK);
    TriadLegacyNode *lb = p->u.program.body[1];
    CHECK(lb->u.loop_block.body_len == 1);
    TriadLegacyNode *iff = lb->u.loop_block.body[0];
    CHECK(iff->kind == TRIAD_LAST_IF_BLOCK);
    CHECK(strcmp(iff->u.if_block.cond_name, "cond") == 0);
    CHECK(iff->u.if_block.has_else);
    CHECK(iff->u.if_block.else_body[0]->kind == TRIAD_LAST_HALT_STMT);
    triad_arena_free(&a);
}

static void test_parse_coupling(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadLegacyNode *p = parse_or_die(&a,
        "couple a -> b kappa = 0.1 for T = 5.0;\n"
        "pair(a, b) kappa = 0.2 for T = 5.0;\n"
        "ring(a, b, c) kappa = 0.05 for T = 2.0;\n"
        "sequence x via (a, b) each_for = 1.0;");
    CHECK(p->u.program.body_len == 4);
    CHECK(p->u.program.body[0]->kind == TRIAD_LAST_COUPLE_STMT);
    CHECK(p->u.program.body[1]->kind == TRIAD_LAST_PAIR_STMT);
    CHECK(p->u.program.body[2]->kind == TRIAD_LAST_RING_STMT);
    CHECK(p->u.program.body[2]->u.ring_seq.members_len == 3);
    CHECK(p->u.program.body[3]->kind == TRIAD_LAST_SEQUENCE_STMT);
    triad_arena_free(&a);
}

static void test_parse_observe_assert_checkpoint(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadLegacyNode *p = parse_or_die(&a,
        "OBSERVE x crystallinity, k_star over_seeds = 3 stream_to \"out.csv\";\n"
        "assert persistent(x);\n"
        "CHECKPOINT x to \"snap.npz\";");
    CHECK(p->u.program.body_len == 3);
    CHECK(p->u.program.body[0]->kind == TRIAD_LAST_OBSERVE_STMT);
    CHECK(p->u.program.body[0]->u.observe_stmt.metrics_len == 2);
    CHECK(p->u.program.body[0]->u.observe_stmt.over_seeds == 3);
    CHECK(strcmp(p->u.program.body[0]->u.observe_stmt.stream_to, "out.csv") == 0);
    CHECK(p->u.program.body[1]->kind == TRIAD_LAST_ASSERT_STMT);
    CHECK(strcmp(p->u.program.body[1]->u.assert_stmt.predicate, "persistent") == 0);
    CHECK(p->u.program.body[2]->kind == TRIAD_LAST_CHECKPOINT_STMT);
    CHECK(strcmp(p->u.program.body[2]->u.checkpoint_stmt.path, "snap.npz") == 0);
    triad_arena_free(&a);
}

static void test_parse_substrate_decl(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadLegacyNode *p = parse_or_die(&a,
        "substrate macro composed_of (a, b, c) { coupling: ring; kappa: 0.1; };");
    CHECK(p->u.program.body_len == 1);
    CHECK(p->u.program.body[0]->kind == TRIAD_LAST_SUBSTRATE_DECL);
    CHECK(p->u.program.body[0]->u.substrate_decl.composed_of_len == 3);
    CHECK(p->u.program.body[0]->u.substrate_decl.properties_len == 2);
    triad_arena_free(&a);
}

int main(void) {
    test_lex_basic();
    test_lex_annot();
    test_lex_hex_binary_neg();
    test_parse_reg_with_regime();
    test_parse_loop_if_else();
    test_parse_coupling();
    test_parse_observe_assert_checkpoint();
    test_parse_substrate_decl();
    if (failures == 0) { printf("test_legacy_frontend: OK\n"); return 0; }
    fprintf(stderr, "test_legacy_frontend: %d FAILURE(S)\n", failures);
    return 1;
}
