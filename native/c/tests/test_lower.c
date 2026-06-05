/* test_lower.c — Native AST→IR lowering self-tests. */
#include "triad_ir.h"

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

static TriadIRNode *lower_or_die(TriadArena *a, const char *src) {
    TriadDiag d = {0};
    TriadAstNode *m = triad_parse_source(a, src, "<t>", &d);
    if (!m) { fprintf(stderr, "parse: %s\n", d.msg ? d.msg : "?"); exit(1); }
    TriadIRNode *ir = triad_ir_lower_module(a, m, &d);
    if (!ir) { fprintf(stderr, "lower failed\n"); exit(1); }
    return ir;
}

static void test_basic_module(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadIRNode *m = lower_or_die(&a, "let x = 1 + 2");
    CHECK(m->kind == TRIAD_IR_MODULE);
    CHECK(m->u.module.body_len == 1);
    TriadIRNode *let = m->u.module.body[0];
    CHECK(let->kind == TRIAD_IR_LET);
    CHECK(strcmp(let->u.let_stmt.name, "x") == 0);
    CHECK(let->u.let_stmt.value->kind == TRIAD_IR_BINOP);
    CHECK(strcmp(let->u.let_stmt.value->u.binop.op, "+") == 0);
    triad_arena_free(&a);
}

static void test_fn_decl(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadIRNode *m = lower_or_die(&a, "fn add(a, b) { return a + b }");
    TriadIRNode *fn = m->u.module.body[0];
    CHECK(fn->kind == TRIAD_IR_FUNCTION);
    CHECK(strcmp(fn->u.func.name, "add") == 0);
    CHECK(fn->u.func.params_len == 2);
    CHECK(strcmp(fn->u.func.params[0], "a") == 0);
    CHECK(fn->u.func.body[0]->kind == TRIAD_IR_RETURN);
    triad_arena_free(&a);
}

static void test_match_lowers_to_if(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    /* match with 3 cases + else → if/elif/elif/else (last elif kept). */
    TriadIRNode *m = lower_or_die(&a,
        "match x {\n"
        "  case 0 => { return 0 }\n"
        "  case 1 => { return 1 }\n"
        "  case 2 => { return 2 }\n"
        "  else => { return -1 }\n"
        "}");
    TriadIRNode *iff = m->u.module.body[0];
    CHECK(iff->kind == TRIAD_IR_IF);
    CHECK(iff->u.if_stmt.condition->kind == TRIAD_IR_BINOP);
    CHECK(strcmp(iff->u.if_stmt.condition->u.binop.op, "==") == 0);
    CHECK(iff->u.if_stmt.elif_clauses_len == 2);
    CHECK(iff->u.if_stmt.has_else);
    triad_arena_free(&a);
}

static void test_match_no_else_promotes_last_elif(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    /* match with 3 cases, no else → if + 1 elif + else (from popped last case). */
    TriadIRNode *m = lower_or_die(&a,
        "match x {\n"
        "  case 0 => { return 0 }\n"
        "  case 1 => { return 1 }\n"
        "  case 2 => { return 2 }\n"
        "}");
    TriadIRNode *iff = m->u.module.body[0];
    CHECK(iff->kind == TRIAD_IR_IF);
    CHECK(iff->u.if_stmt.elif_clauses_len == 1);
    CHECK(iff->u.if_stmt.has_else);
    CHECK(iff->u.if_stmt.else_body_len == 1);
    triad_arena_free(&a);
}

static void test_match_wildcard(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    /* `case _` becomes IRBool(True) condition, not a comparison. */
    TriadIRNode *m = lower_or_die(&a,
        "match x {\n"
        "  case 0 => { return 0 }\n"
        "  case _ => { return -1 }\n"
        "}");
    TriadIRNode *iff = m->u.module.body[0];
    CHECK(iff->kind == TRIAD_IR_IF);
    /* No elif: the wildcard becomes the else_body via the pop rule. */
    CHECK(iff->u.if_stmt.elif_clauses_len == 0);
    CHECK(iff->u.if_stmt.has_else);
    triad_arena_free(&a);
}

static void test_class_lowers_to_class_decl(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadIRNode *m = lower_or_die(&a,
        "class Point { x: int; y: int; fn norm() { return self.x*self.x + self.y*self.y } }");
    TriadIRNode *cd = m->u.module.body[0];
    CHECK(cd->kind == TRIAD_IR_CLASS_DECL);
    CHECK(strcmp(cd->u.class_decl.name, "Point") == 0);
    CHECK(cd->u.class_decl.fields_len == 2);
    CHECK(cd->u.class_decl.methods_len == 1);
    CHECK(cd->u.class_decl.methods[0]->kind == TRIAD_IR_FUNCTION);
    triad_arena_free(&a);
}

static void test_from_import_drops_names(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    /* `from triad import solve, regime` → IRImport(path=[triad], alias=None). */
    TriadIRNode *m = lower_or_die(&a, "from triad import solve, regime");
    TriadIRNode *im = m->u.module.body[0];
    CHECK(im->kind == TRIAD_IR_IMPORT);
    CHECK(im->u.import_stmt.path_len == 1);
    CHECK(strcmp(im->u.import_stmt.path[0], "triad") == 0);
    CHECK(im->u.import_stmt.alias == NULL);
    triad_arena_free(&a);
}

static void test_slice_index_expression(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadIRNode *m = lower_or_die(&a, "let y = xs[1:4:2]");
    TriadIRNode *let = m->u.module.body[0];
    TriadIRNode *sl = let->u.let_stmt.value;
    CHECK(sl->kind == TRIAD_IR_SLICE);
    CHECK(sl->u.slice.start != NULL);
    CHECK(sl->u.slice.end   != NULL);
    CHECK(sl->u.slice.step  != NULL);
    triad_arena_free(&a);
}

static void test_fstring_parts_lower(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    /* IRFString.parts uses ("", expr) for interpolations and (text, None) for literals. */
    TriadIRNode *m = lower_or_die(&a, "print(f\"x={n + 1}\")");
    TriadIRNode *call = m->u.module.body[0]->u.expr_stmt.expr;
    CHECK(call->kind == TRIAD_IR_CALL);
    TriadIRNode *fs = call->u.call.args[0];
    CHECK(fs->kind == TRIAD_IR_FSTRING);
    CHECK(fs->u.fstring.parts_len == 2);
    CHECK(fs->u.fstring.parts[0].expr == NULL);
    CHECK(strcmp(fs->u.fstring.parts[0].text, "x=") == 0);
    CHECK(fs->u.fstring.parts[1].expr != NULL);
    CHECK(fs->u.fstring.parts[1].expr->kind == TRIAD_IR_BINOP);
    triad_arena_free(&a);
}

static void test_reg_carries_regime(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    /* Substrate declaration must lower into IRRegDecl with regime/value;
     * the overrides on the AST are intentionally dropped (Python parity). */
    TriadIRNode *m = lower_or_die(&a, "reg s: anti_collapse = 1.0");
    TriadIRNode *r = m->u.module.body[0];
    CHECK(r->kind == TRIAD_IR_REG_DECL);
    CHECK(strcmp(r->u.reg_decl.name, "s") == 0);
    CHECK(strcmp(r->u.reg_decl.regime, "anti_collapse") == 0);
    CHECK(r->u.reg_decl.value != NULL);
    triad_arena_free(&a);
}

int main(void) {
    test_basic_module();
    test_fn_decl();
    test_match_lowers_to_if();
    test_match_no_else_promotes_last_elif();
    test_match_wildcard();
    test_class_lowers_to_class_decl();
    test_from_import_drops_names();
    test_slice_index_expression();
    test_fstring_parts_lower();
    test_reg_carries_regime();
    if (failures == 0) { printf("test_lower: OK\n"); return 0; }
    fprintf(stderr, "test_lower: %d FAILURE(S)\n", failures);
    return 1;
}
