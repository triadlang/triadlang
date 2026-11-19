/*
 * TriadLang Native Runtime — Stdlib wrappers (math, random, io, range)
 */
#include "triad_rt.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ── Print ── */

void triad_print(int32_t nargs, TriadValue *args) {
    for (int32_t i = 0; i < nargs; i++) {
        if (i > 0) printf(" ");
        TriadString *s = triad_value_to_string(args[i]);
        printf("%s", s->data);
        triad_str_free(s);
    }
    printf("\n");
}

TriadValue triad_input(TriadString *prompt) {
    if (prompt && prompt->len > 0) {
        printf("%s", prompt->data);
        fflush(stdout);
    }
    char buf[4096];
    if (!fgets(buf, sizeof(buf), stdin)) return TRIAD_NONE_VAL;
    int32_t len = (int32_t)strlen(buf);
    if (len > 0 && buf[len - 1] == '\n') len--;
    TriadString *s = triad_str_new_len(buf, len);
    return (TriadValue){.tag = TRIAD_STRING, .as = {.sval = s}};
}

/* ── Math ── */

double triad_math_sqrt(double x)    { return sqrt(x); }
double triad_math_sin(double x)     { return sin(x); }
double triad_math_cos(double x)     { return cos(x); }
double triad_math_tan(double x)     { return tan(x); }
double triad_math_exp(double x)     { return exp(x); }
double triad_math_log(double x)     { return log(x); }
double triad_math_log10(double x)   { return log10(x); }
double triad_math_floor(double x)   { return floor(x); }
double triad_math_ceil(double x)    { return ceil(x); }
double triad_math_abs(double x)     { return fabs(x); }
int64_t triad_math_abs_int(int64_t x) { return x < 0 ? -x : x; }
int64_t triad_math_clamp(int64_t x, int64_t lo, int64_t hi) { return x < lo ? lo : (x > hi ? hi : x); }
double triad_math_pow(double b, double e) { return pow(b, e); }
int64_t triad_math_min(int64_t a, int64_t b) { return a < b ? a : b; }
int64_t triad_math_max(int64_t a, int64_t b) { return a > b ? a : b; }

/* ── Random ── */

static uint64_t _xorshift64(void) {
    static uint64_t state = 0;
    if (state == 0) state = (uint64_t)time(NULL) ^ 0xDEADBEEFCAFE1234ULL;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

void triad_random_seed(uint64_t seed) {
    srand((unsigned int)seed);
}

double triad_random_double(void) {
    return (double)_xorshift64() / (double)UINT64_MAX;
}

int64_t triad_random_int(int64_t lo, int64_t hi) {
    return lo + (int64_t)(_xorshift64() % (uint64_t)(hi - lo + 1));
}

TriadValue triad_random_choice(TriadList *l) {
    if (!l || l->len == 0) return TRIAD_NONE_VAL;
    int32_t idx = (int32_t)(_xorshift64() % (uint64_t)l->len);
    return triad_list_get(l, idx);
}

void triad_random_shuffle(TriadList *l) {
    if (!l || l->len < 2) return;
    for (int32_t i = l->len - 1; i > 0; i--) {
        int32_t j = (int32_t)(_xorshift64() % (uint64_t)(i + 1));
        TriadValue tmp = l->items[i];
        l->items[i] = l->items[j];
        l->items[j] = tmp;
    }
}

double triad_random_uniform(double lo, double hi) {
    return lo + triad_random_double() * (hi - lo);
}

/* ── Range ── */

TriadList *triad_range(int64_t start, int64_t stop, int64_t step) {
    TriadList *l = triad_list_new();
    if (step == 0) return l;
    if (step > 0) {
        for (int64_t i = start; i < stop; i += step)
            triad_list_push(l, TRIAD_INT(i));
    } else {
        for (int64_t i = start; i > stop; i += step)
            triad_list_push(l, TRIAD_INT(i));
    }
    return l;
}
