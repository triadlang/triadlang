/* triad_check_legacy.c — Port of compiler/typecheck.py (v1 DSL).
 *
 * Walks a parsed legacy Program once, accumulates every error, and
 * returns 0 on clean check, -1 otherwise. The error text format is
 * byte-identical to Python's TypeCheckError messages so that
 * scripts/typecheck_legacy_to_text.py and parity_check_legacy can be
 * diff'd line-by-line over the crossdomain .tri fixtures.
 *
 * Ownership: error strings are pushed into the supplied arena. Lists
 * of declared/regime/metric/projection names are formatted with the
 * same repr() convention Python uses (single quotes, ", " separator).
 */
#include "triad_check_legacy.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ────────────────────────────────────────────────────────────────────
 * Known sets — must match compiler/typecheck.py exactly.
 * Ordering does not matter for membership; the formatter sorts them.
 * ──────────────────────────────────────────────────────────────────── */

static const char *KNOWN_METRICS[] = {
    "k_star", "IPR", "crystallinity", "peak", "FWHM", "norm",
    "stabilization", "memory_persistence",
    "bravais_family", "cluster_count", "cluster_centroids",
    "slow_state", "slow_state_late_mean",
    "LZc", "metastability",
    "phi_id", "pcist", "causal_density", "kuramoto",
    "atom_count", "atom_centroids", "atom_separation",
    "atom_persistence_late",
    "atoms_per_region",
    "atomicity_ratio",
};
#define KNOWN_METRICS_LEN (sizeof(KNOWN_METRICS) / sizeof(KNOWN_METRICS[0]))

static const char *KNOWN_PREDICATES[] = {
    "persistent", "extended", "structurally_open", "non_trivial_memory",
    "atomic", "anti_collapsed",
};
#define KNOWN_PREDICATES_LEN (sizeof(KNOWN_PREDICATES) / sizeof(KNOWN_PREDICATES[0]))

static const char *KNOWN_PROJECTIONS[] = {
    "centroid_density", "integrated_density", "dominant_k_star",
    "fourier_band", "atoms_of_atoms", "none",
};
#define KNOWN_PROJECTIONS_LEN (sizeof(KNOWN_PROJECTIONS) / sizeof(KNOWN_PROJECTIONS[0]))

/* ────────────────────────────────────────────────────────────────────
 * Tiny dynamic string set (linear-probe). Keys are non-owned cstrings.
 * ──────────────────────────────────────────────────────────────────── */

typedef struct {
    const char **buckets;
    size_t       cap;
    size_t       len;
} StrSet;

static size_t fnv1a(const char *s) {
    size_t h = 1469598103934665603ULL;
    for (; *s; ++s) {
        h ^= (unsigned char)*s;
        h *= 1099511628211ULL;
    }
    return h;
}

static void ss_init(StrSet *s, size_t cap) {
    if (cap < 16) cap = 16;
    s->cap = cap;
    s->len = 0;
    s->buckets = (const char **)calloc(cap, sizeof(char *));
}

static int ss_has(const StrSet *s, const char *k) {
    if (!s->cap) return 0;
    size_t h = fnv1a(k) % s->cap;
    while (s->buckets[h]) {
        if (strcmp(s->buckets[h], k) == 0) return 1;
        h = (h + 1) % s->cap;
    }
    return 0;
}

static void ss_rehash(StrSet *s, size_t new_cap);

static void ss_add(StrSet *s, const char *k) {
    if ((s->len + 1) * 2 > s->cap) ss_rehash(s, s->cap * 2);
    size_t h = fnv1a(k) % s->cap;
    while (s->buckets[h]) {
        if (strcmp(s->buckets[h], k) == 0) return;
        h = (h + 1) % s->cap;
    }
    s->buckets[h] = k;
    s->len++;
}

static void ss_rehash(StrSet *s, size_t new_cap) {
    const char **old = s->buckets;
    size_t old_cap = s->cap;
    s->buckets = (const char **)calloc(new_cap, sizeof(char *));
    s->cap = new_cap;
    s->len = 0;
    for (size_t i = 0; i < old_cap; ++i) {
        if (old[i]) ss_add(s, old[i]);
    }
    free(old);
}

static void ss_free(StrSet *s) {
    free(s->buckets);
    s->buckets = NULL;
    s->cap = s->len = 0;
}

/* Materialize as a sorted array (lexicographic, Python's sorted()). */
static void ss_sorted(const StrSet *s, const char ***out, size_t *out_len) {
    const char **arr = (const char **)malloc(sizeof(char *) * (s->len + 1));
    size_t n = 0;
    for (size_t i = 0; i < s->cap; ++i)
        if (s->buckets[i]) arr[n++] = s->buckets[i];
    /* qsort by strcmp */
    for (size_t i = 1; i < n; ++i) {
        const char *cur = arr[i];
        size_t j = i;
        while (j > 0 && strcmp(arr[j - 1], cur) > 0) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = cur;
    }
    *out = arr;
    *out_len = n;
}

/* ────────────────────────────────────────────────────────────────────
 * Error accumulator.
 * ──────────────────────────────────────────────────────────────────── */

typedef struct {
    const char **items;
    size_t       len;
    size_t       cap;
    TriadArena  *arena;
} ErrBuf;

static void eb_init(ErrBuf *e, TriadArena *arena) {
    e->items = NULL;
    e->len = e->cap = 0;
    e->arena = arena;
}

static void eb_push_owned(ErrBuf *e, const char *owned) {
    if (e->len + 1 > e->cap) {
        size_t nc = e->cap ? e->cap * 2 : 16;
        e->items = (const char **)realloc(e->items, sizeof(char *) * nc);
        e->cap = nc;
    }
    e->items[e->len++] = owned;
}

/* Allocate a copy in the arena. */
static const char *arena_strdup(TriadArena *a, const char *s) {
    size_t n = strlen(s) + 1;
    char *dst = (char *)triad_arena_alloc(a, n);
    memcpy(dst, s, n);
    return dst;
}

/* vprintf into a freshly-allocated buffer, then copy into arena. */
static const char *fmt_arena(TriadArena *a, const char *fmt, ...) {
    va_list ap1, ap2;
    va_start(ap1, fmt);
    va_copy(ap2, ap1);
    int need = vsnprintf(NULL, 0, fmt, ap1);
    va_end(ap1);
    if (need < 0) { va_end(ap2); return arena_strdup(a, ""); }
    char *buf = (char *)malloc((size_t)need + 1);
    vsnprintf(buf, (size_t)need + 1, fmt, ap2);
    va_end(ap2);
    const char *out = arena_strdup(a, buf);
    free(buf);
    return out;
}

/* ────────────────────────────────────────────────────────────────────
 * Python-style list formatting:
 *   ['a', 'b', 'c']           (no truncation marker)
 *   ['a', 'b', 'c', 'd', 'e'] (when sliced with no marker)
 *   ['a', 'b', 'c']...        (when caller wants the trailing "..."
 *                              suffix appended outside the brackets)
 *
 * format_pylist returns the bracketed list only. The "..." suffix is
 * appended by the caller when needed, matching Python's
 *   f"{sorted(s)[:K]}{'...' if len(s) > K else ''}"
 * behavior precisely.
 * ──────────────────────────────────────────────────────────────────── */

static const char *format_pylist(TriadArena *a,
                                 const char *const *items,
                                 size_t                len) {
    /* Compute exact size. */
    size_t total = 2;            /* [ ] */
    for (size_t i = 0; i < len; ++i) {
        total += 2;              /* surrounding quotes */
        total += strlen(items[i]);
        if (i + 1 < len) total += 2;  /* ", " */
    }
    char *buf = (char *)malloc(total + 1);
    size_t pos = 0;
    buf[pos++] = '[';
    for (size_t i = 0; i < len; ++i) {
        buf[pos++] = '\'';
        size_t L = strlen(items[i]);
        memcpy(buf + pos, items[i], L);
        pos += L;
        buf[pos++] = '\'';
        if (i + 1 < len) { buf[pos++] = ','; buf[pos++] = ' '; }
    }
    buf[pos++] = ']';
    buf[pos] = '\0';
    const char *out = arena_strdup(a, buf);
    free(buf);
    return out;
}

/* Mirror sorted(set)[:K]{'...' if len > K else ''}. */
static const char *sorted_slice(TriadArena *a,
                                const StrSet *s,
                                size_t k_limit,
                                int always_dots) {
    const char **arr = NULL;
    size_t n = 0;
    ss_sorted(s, &arr, &n);
    size_t take = (n < k_limit) ? n : k_limit;
    const char *body = format_pylist(a, arr, take);
    int add_dots = always_dots || (n > k_limit);
    const char *out;
    if (add_dots) {
        out = fmt_arena(a, "%s...", body);
    } else {
        out = body;
    }
    free(arr);
    return out;
}

/* Same for a flat string array (used by KNOWN_METRICS etc.). */
static const char *sorted_slice_arr(TriadArena *a,
                                    const char *const *items,
                                    size_t                items_len,
                                    size_t                k_limit,
                                    int                   always_dots) {
    StrSet tmp;
    ss_init(&tmp, items_len * 2 + 16);
    for (size_t i = 0; i < items_len; ++i) ss_add(&tmp, items[i]);
    const char *out = sorted_slice(a, &tmp, k_limit, always_dots);
    ss_free(&tmp);
    return out;
}

/* Sorted list of all items (no slicing). Used for projections. */
static const char *sorted_full_arr(TriadArena *a,
                                   const char *const *items,
                                   size_t                items_len) {
    StrSet tmp;
    ss_init(&tmp, items_len * 2 + 16);
    for (size_t i = 0; i < items_len; ++i) ss_add(&tmp, items[i]);
    const char **arr = NULL;
    size_t n = 0;
    ss_sorted(&tmp, &arr, &n);
    const char *out = format_pylist(a, arr, n);
    free(arr);
    ss_free(&tmp);
    return out;
}

/* ────────────────────────────────────────────────────────────────────
 * Helpers used inside walk_body.
 * ──────────────────────────────────────────────────────────────────── */

static int is_known_metric(const char *name) {
    for (size_t i = 0; i < KNOWN_METRICS_LEN; ++i)
        if (strcmp(KNOWN_METRICS[i], name) == 0) return 1;
    return 0;
}

static int is_known_predicate(const char *name) {
    for (size_t i = 0; i < KNOWN_PREDICATES_LEN; ++i)
        if (strcmp(KNOWN_PREDICATES[i], name) == 0) return 1;
    return 0;
}

static int is_known_projection(const char *name) {
    for (size_t i = 0; i < KNOWN_PROJECTIONS_LEN; ++i)
        if (strcmp(KNOWN_PROJECTIONS[i], name) == 0) return 1;
    return 0;
}

static int regime_known(const char *const *regs, size_t n, const char *name) {
    for (size_t i = 0; i < n; ++i)
        if (strcmp(regs[i], name) == 0) return 1;
    return 0;
}

static const char *unknown_hint(TriadArena *a, const StrSet *declared) {
    return sorted_slice(a, declared, 8, /*always_dots=*/0);
}

/* ────────────────────────────────────────────────────────────────────
 * Walk.
 * ──────────────────────────────────────────────────────────────────── */

typedef struct {
    StrSet            *declared;
    const char *const *regime_names;
    size_t             regime_count;
    ErrBuf            *errs;
    TriadArena        *arena;
} Ctx;

static void check_known(Ctx *c, const char *name, int line, const char *what) {
    if (!ss_has(c->declared, name)) {
        const char *hint = unknown_hint(c->arena, c->declared);
        eb_push_owned(c->errs,
            fmt_arena(c->arena,
                "L%d: unknown %s '%s' (declared: %s)",
                line, what, name, hint));
    }
}

static void check_regime(Ctx *c, const char *regime, int line, const char *macro) {
    if (!regime) return;
    if (regime_known(c->regime_names, c->regime_count, regime)) return;
    if (macro) {
        eb_push_owned(c->errs,
            fmt_arena(c->arena,
                "L%d: unknown macro regime '%s'", line, regime));
    } else {
        const char *hint;
        if (c->regime_count == 0) {
            hint = "[]";
        } else {
            hint = sorted_slice_arr(c->arena, c->regime_names,
                                    c->regime_count, 5,
                                    /*always_dots=*/1);
        }
        eb_push_owned(c->errs,
            fmt_arena(c->arena,
                "L%d: unknown regime '%s' (available: %s)",
                line, regime, hint));
    }
}

static void walk_body(Ctx *c, TriadLegacyNode *const *body, size_t n);

static void walk_stmt(Ctx *c, const TriadLegacyNode *stmt) {
    switch (stmt->kind) {
    case TRIAD_LAST_ANNOTATION:
        break;

    case TRIAD_LAST_REG_DECL: {
        const char *name = stmt->u.reg_decl.name;
        int line = stmt->line;
        if (ss_has(c->declared, name)) {
            eb_push_owned(c->errs,
                fmt_arena(c->arena,
                    "L%d: duplicate substrate '%s'", line, name));
        } else {
            ss_add(c->declared, name);
        }
        if (stmt->u.reg_decl.regime_name) {
            check_regime(c, stmt->u.reg_decl.regime_name, line, NULL);
        }
        break;
    }

    case TRIAD_LAST_EVOLVE_STMT: {
        check_known(c, stmt->u.evolve_stmt.target, stmt->line, "substrate");
        break;
    }

    case TRIAD_LAST_COUPLE_STMT: {
        check_known(c, stmt->u.couple_pair.src_or_a, stmt->line, "couple src");
        check_known(c, stmt->u.couple_pair.dst_or_b, stmt->line, "couple dst");
        break;
    }

    case TRIAD_LAST_PAIR_STMT: {
        check_known(c, stmt->u.couple_pair.src_or_a, stmt->line, "pair a");
        check_known(c, stmt->u.couple_pair.dst_or_b, stmt->line, "pair b");
        break;
    }

    case TRIAD_LAST_RING_STMT: {
        for (size_t i = 0; i < stmt->u.ring_seq.members_len; ++i) {
            check_known(c, stmt->u.ring_seq.members[i], stmt->line, "ring member");
        }
        break;
    }

    case TRIAD_LAST_SEQUENCE_STMT: {
        check_known(c, stmt->u.ring_seq.target, stmt->line, "sequence target");
        for (size_t i = 0; i < stmt->u.ring_seq.members_len; ++i) {
            check_known(c, stmt->u.ring_seq.members[i], stmt->line, "sequence input");
        }
        break;
    }

    case TRIAD_LAST_OBSERVE_STMT: {
        check_known(c, stmt->u.observe_stmt.target, stmt->line, "OBSERVE");
        for (size_t i = 0; i < stmt->u.observe_stmt.metrics_len; ++i) {
            const char *m = stmt->u.observe_stmt.metrics[i];
            if (!is_known_metric(m)) {
                const char *hint = sorted_slice_arr(c->arena, KNOWN_METRICS,
                                                    KNOWN_METRICS_LEN, 5,
                                                    /*always_dots=*/1);
                eb_push_owned(c->errs,
                    fmt_arena(c->arena,
                        "L%d: unknown OBSERVE metric '%s' (known: %s)",
                        stmt->line, m, hint));
            }
        }
        break;
    }

    case TRIAD_LAST_ASSERT_STMT: {
        /* Python uses the literal what="assert" for the target lookup,
         * which produces e.g. "unknown assert 'zzz'" (not "unknown
         * substrate"). Mirror that. */
        check_known(c, stmt->u.assert_stmt.target, stmt->line, "assert");
        if (!is_known_predicate(stmt->u.assert_stmt.predicate)) {
            eb_push_owned(c->errs,
                fmt_arena(c->arena,
                    "L%d: unknown assert predicate '%s'",
                    stmt->line, stmt->u.assert_stmt.predicate));
        }
        break;
    }

    case TRIAD_LAST_SUBSTRATE_DECL: {
        const char *name = stmt->u.substrate_decl.name;
        int line = stmt->line;
        for (size_t i = 0; i < stmt->u.substrate_decl.composed_of_len; ++i) {
            check_known(c, stmt->u.substrate_decl.composed_of[i], line,
                        "composed_of member");
        }
        if (ss_has(c->declared, name)) {
            eb_push_owned(c->errs,
                fmt_arena(c->arena,
                    "L%d: duplicate substrate '%s'", line, name));
        } else {
            ss_add(c->declared, name);
        }
        const char *projection = "none";
        const char *regime = NULL;
        for (size_t i = 0; i < stmt->u.substrate_decl.properties_len; ++i) {
            const TriadLegacyOverride *p = &stmt->u.substrate_decl.properties[i];
            if (strcmp(p->key, "projection") == 0) {
                if (p->value.kind == TRIAD_LVAL_STR ||
                    p->value.kind == TRIAD_LVAL_IDENT) {
                    projection = p->value.str_val;
                }
            } else if (strcmp(p->key, "regime") == 0) {
                if (p->value.kind == TRIAD_LVAL_STR ||
                    p->value.kind == TRIAD_LVAL_IDENT) {
                    regime = p->value.str_val;
                }
            }
        }
        if (!is_known_projection(projection)) {
            const char *full = sorted_full_arr(c->arena, KNOWN_PROJECTIONS,
                                               KNOWN_PROJECTIONS_LEN);
            eb_push_owned(c->errs,
                fmt_arena(c->arena,
                    "L%d: unknown projection '%s' (known: %s)",
                    line, projection, full));
        }
        if (regime) check_regime(c, regime, line, /*macro=*/"macro");
        break;
    }

    case TRIAD_LAST_OP:
    case TRIAD_LAST_OUT_STMT:
    case TRIAD_LAST_HALT_STMT:
    case TRIAD_LAST_CHECKPOINT_STMT:
        /* legacy imperative — typecheck delegated to compiler */
        break;

    case TRIAD_LAST_LOOP_BLOCK:
        walk_body(c, stmt->u.loop_block.body, stmt->u.loop_block.body_len);
        break;

    case TRIAD_LAST_SEGMENT_BLOCK:
        walk_body(c, stmt->u.segment_block.body, stmt->u.segment_block.body_len);
        break;

    case TRIAD_LAST_IF_BLOCK:
        walk_body(c, stmt->u.if_block.then_body, stmt->u.if_block.then_body_len);
        if (stmt->u.if_block.has_else)
            walk_body(c, stmt->u.if_block.else_body, stmt->u.if_block.else_body_len);
        break;

    default:
        /* literals / refs / Program — nothing to check at top level */
        break;
    }
}

static void walk_body(Ctx *c, TriadLegacyNode *const *body, size_t n) {
    for (size_t i = 0; i < n; ++i) walk_stmt(c, body[i]);
}

/* ────────────────────────────────────────────────────────────────────
 * Public entry point.
 * ──────────────────────────────────────────────────────────────────── */

int triad_check_legacy_program(TriadArena              *arena,
                               const TriadLegacyNode   *program,
                               const char *const       *regime_names,
                               size_t                   regime_count,
                               TriadCheckLegacyErrors  *out) {
    if (!program || program->kind != TRIAD_LAST_PROGRAM) return -1;

    StrSet declared;
    ss_init(&declared, 32);
    /* Pre-declare the constants the legacy DSL ships with. */
    ss_add(&declared, "ZERO");
    ss_add(&declared, "ONE");
    ss_add(&declared, "TWO");
    ss_add(&declared, "NEG_ONE");

    ErrBuf errs;
    eb_init(&errs, arena);

    Ctx c = {
        .declared = &declared,
        .regime_names = regime_names,
        .regime_count = regime_count,
        .errs = &errs,
        .arena = arena,
    };

    walk_body(&c, program->u.program.body, program->u.program.body_len);

    int rc = errs.len ? -1 : 0;
    if (out) {
        out->items = errs.items;
        out->len = errs.len;
    } else {
        free(errs.items);
    }
    ss_free(&declared);
    return rc;
}
