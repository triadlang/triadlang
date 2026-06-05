/* test_check.c — Native typecheck unit tests. */
#include "triad_check.h"

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

static int run_check(TriadArena *a, const char *src, TriadCheckErrors *errs) {
    TriadDiag d = {0};
    TriadAstNode *m = triad_parse_source(a, src, "<t>", &d);
    if (!m) { fprintf(stderr, "parse: %s\n", d.msg ? d.msg : "?"); exit(1); }
    return triad_check_module(a, m, errs);
}

static void test_undeclared_var(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadCheckErrors errs = {0};
    int rc = run_check(&a, "print(notdef)", &errs);
    CHECK(rc != 0);
    CHECK(errs.len == 1);
    CHECK(strstr(errs.items[0], "E2001") != NULL);
    CHECK(strstr(errs.items[0], "notdef") != NULL);
    triad_arena_free(&a);
}

static void test_builtin_arity(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadCheckErrors errs = {0};
    int rc = run_check(&a, "let x = len(1, 2)", &errs);
    CHECK(rc != 0);
    CHECK(errs.len == 1);
    CHECK(strstr(errs.items[0], "E2002") != NULL);
    CHECK(strstr(errs.items[0], "'len'") != NULL);
    triad_arena_free(&a);
}

static void test_user_fn_arity(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadCheckErrors errs = {0};
    int rc = run_check(&a,
        "fn add(a, b) { return a + b }\n"
        "let r = add(1)\n", &errs);
    CHECK(rc != 0);
    CHECK(errs.len == 1);
    CHECK(strstr(errs.items[0], "E2002") != NULL);
    CHECK(strstr(errs.items[0], "'add'") != NULL);
    triad_arena_free(&a);
}

static void test_self_is_implicit(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadCheckErrors errs = {0};
    int rc = run_check(&a, "type T { x: int; fn get() { return self.x } }", &errs);
    CHECK(rc == 0);
    CHECK(errs.len == 0);
    triad_arena_free(&a);
}

static void test_undeclared_substrate(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadCheckErrors errs = {0};
    int rc = run_check(&a, "couple a -> b kappa = 0.1", &errs);
    CHECK(rc != 0);
    CHECK(errs.len == 2);
    CHECK(strstr(errs.items[0], "E2011") != NULL);
    triad_arena_free(&a);
}

static void test_observe_unknown_target(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadCheckErrors errs = {0};
    int rc = run_check(&a, "observe ghost crystallinity", &errs);
    CHECK(rc != 0);
    CHECK(errs.len == 1);
    CHECK(strstr(errs.items[0], "E2010") != NULL);
    CHECK(strstr(errs.items[0], "ghost") != NULL);
    triad_arena_free(&a);
}

static void test_observe_known_target(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadCheckErrors errs = {0};
    int rc = run_check(&a, "reg s\nobserve s crystallinity", &errs);
    CHECK(rc == 0);
    triad_arena_free(&a);
}

static void test_for_var_in_scope(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadCheckErrors errs = {0};
    int rc = run_check(&a,
        "let xs = [1, 2, 3]\n"
        "for x in xs { print(x) }\n", &errs);
    CHECK(rc == 0);
    triad_arena_free(&a);
}

static void test_lambda_params_in_scope(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadCheckErrors errs = {0};
    int rc = run_check(&a, "let f = fn(a, b) { return a + b }", &errs);
    CHECK(rc == 0);
    triad_arena_free(&a);
}

static void test_fstring_expr_checked(void) {
    TriadArena a; triad_arena_init(&a, 4096);
    TriadCheckErrors errs = {0};
    int rc = run_check(&a, "print(f\"x={zzz}\")", &errs);
    CHECK(rc != 0);
    CHECK(strstr(errs.items[0], "E2001") != NULL);
    CHECK(strstr(errs.items[0], "zzz") != NULL);
    triad_arena_free(&a);
}

static void test_clean_demo(void) {
    /* Sanity: the demo example must check clean. */
    TriadArena a; triad_arena_init(&a, 4096);
    TriadCheckErrors errs = {0};
    int rc = run_check(&a,
        "fn factorial(n) {\n"
        "    if n <= 1 { return 1 }\n"
        "    return n * factorial(n - 1)\n"
        "}\n"
        "let r = factorial(10)\n"
        "print(f\"10! = {r}\")\n", &errs);
    CHECK(rc == 0);
    triad_arena_free(&a);
}

int main(void) {
    test_undeclared_var();
    test_builtin_arity();
    test_user_fn_arity();
    test_self_is_implicit();
    test_undeclared_substrate();
    test_observe_unknown_target();
    test_observe_known_target();
    test_for_var_in_scope();
    test_lambda_params_in_scope();
    test_fstring_expr_checked();
    test_clean_demo();
    if (failures == 0) { printf("test_check: OK\n"); return 0; }
    fprintf(stderr, "test_check: %d FAILURE(S)\n", failures);
    return 1;
}
