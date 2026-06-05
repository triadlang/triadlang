/*
 * Smoke test for TriadLang C runtime
 */
#include "triad_rt.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    printf("=== TriadLang C Runtime Smoke Test ===\n\n");

    printf("--- Primitives ---\n");
    TriadValue none = TRIAD_NONE_VAL;
    TriadValue b = TRIAD_BOOL(true);
    TriadValue i = TRIAD_INT(42);
    TriadValue f = TRIAD_FLOAT(3.14);
    assert(!triad_is_truthy(none));
    assert(triad_is_truthy(b));
    assert(triad_is_truthy(i));
    assert(triad_is_truthy(f));
    printf("  Primitives: OK\n");

    printf("--- Strings ---\n");
    TriadString *s1 = triad_str_new("hello");
    TriadString *s2 = triad_str_new(" world");
    TriadString *s3 = triad_str_concat(s1, s2);
    assert(triad_str_eq(s3, triad_str_new("hello world")));
    assert(triad_str_len(s3) == 11);
    TriadString *upper = triad_str_upper(s1);
    assert(triad_str_eq(upper, triad_str_new("HELLO")));
    TriadString *sl = triad_str_slice(s3, 0, 5, 1);
    assert(triad_str_eq(sl, triad_str_new("hello")));
    printf("  concat: '%s' (len=%d)\n", s3->data, s3->len);
    printf("  upper: '%s'\n", upper->data);
    triad_str_free(s1); triad_str_free(s2); triad_str_free(s3);
    triad_str_free(upper); triad_str_free(sl);
    printf("  Strings: OK\n");

    printf("--- Lists ---\n");
    TriadList *list = triad_list_new();
    triad_list_push(list, TRIAD_INT(10));
    triad_list_push(list, TRIAD_INT(20));
    triad_list_push(list, TRIAD_INT(30));
    assert(triad_list_len(list) == 3);
    assert(triad_list_get(list, 0).as.ival == 10);
    assert(triad_list_get(list, 2).as.ival == 30);
    assert(triad_list_contains(list, TRIAD_INT(20)));
    assert(!triad_list_contains(list, TRIAD_INT(99)));
    TriadList *sliced = triad_list_slice(list, 1, 3, 1);
    assert(triad_list_len(sliced) == 2);
    assert(triad_list_get(sliced, 0).as.ival == 20);
    TriadList *rev = triad_list_reversed(list);
    assert(triad_list_get(rev, 0).as.ival == 30);
    triad_list_insert(list, 1, TRIAD_INT(15));
    assert(triad_list_len(list) == 4);
    assert(triad_list_get(list, 1).as.ival == 15);
    triad_list_remove_at(list, 1);
    assert(triad_list_len(list) == 3);
    printf("  Lists: OK\n");

    printf("--- Dict ---\n");
    TriadDict *dict = triad_dict_new();
    TriadString *k1 = triad_str_new("name");
    TriadString *k2 = triad_str_new("age");
    triad_dict_set(dict, k1, (TriadValue){.tag = TRIAD_STRING, .as = {.sval = triad_str_new("TriadLang")}});
    triad_dict_set(dict, k2, TRIAD_INT(1));
    assert(triad_dict_len(dict) == 2);
    assert(triad_dict_has(dict, k1));
    TriadValue v = triad_dict_get(dict, k2);
    assert(v.tag == TRIAD_INT && v.as.ival == 1);
    triad_dict_del(dict, k2);
    assert(triad_dict_len(dict) == 1);
    printf("  Dict: OK\n");

    printf("--- NDArray ---\n");
    int32_t shape2d[] = {2, 3};
    TriadNDArray *arr = triad_ndarray_zeros(2, shape2d);
    assert(triad_ndarray_size(arr) == 6);
    int32_t idx00[] = {0, 0};
    triad_ndarray_set(arr, idx00, 42.0);
    assert(triad_ndarray_get(arr, idx00) == 42.0);
    int32_t idx11[] = {1, 1};
    triad_ndarray_set(arr, idx11, 7.0);
    assert(triad_ndarray_sum(arr) == 49.0);
    assert(triad_ndarray_mean(arr) == 49.0 / 6.0);
    TriadNDArray *ones = triad_ndarray_ones(2, shape2d);
    assert(triad_ndarray_sum(ones) == 6.0);
    printf("  NDArray: OK\n");

    printf("--- Range ---\n");
    TriadList *r = triad_range(0, 5, 1);
    assert(triad_list_len(r) == 5);
    assert(triad_list_get(r, 0).as.ival == 0);
    assert(triad_list_get(r, 4).as.ival == 4);
    TriadList *r2 = triad_range(10, 0, -2);
    assert(triad_list_len(r2) == 5);
    assert(triad_list_get(r2, 0).as.ival == 10);
    printf("  Range: OK\n");

    printf("--- Print ---\n");
    TriadValue pargs[] = { TRIAD_INT(2026), (TriadValue){.tag = TRIAD_STRING, .as = {.sval = triad_str_new("TriadLang native runtime OK")}} };
    triad_print(2, pargs);

    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
