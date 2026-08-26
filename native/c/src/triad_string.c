#include "triad_rt.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>

TriadString *triad_str_new(const char *data) {
    return triad_str_new_len(data, data ? (int32_t)strlen(data) : 0);
}

TriadString *triad_str_new_len(const char *data, int32_t len) {
    TriadString *s = malloc(sizeof(TriadString));
    s->refcount = 1;
    s->len = len;
    s->cap = len < 16 ? 16 : len + 1;
    s->data = malloc(s->cap);
    if (data && len > 0) memcpy(s->data, data, len);
    s->data[len] = '\0';
    return s;
}

TriadString *triad_str_copy(TriadString *s) {
    if (!s) return triad_str_new("");
    return triad_str_new_len(s->data, s->len);
}

void triad_str_free(TriadString *s) {
    if (!s) return;
    free(s->data);
    free(s);
}

TriadString *triad_str_concat(TriadString *a, TriadString *b) {
    int32_t len = (a ? a->len : 0) + (b ? b->len : 0);
    TriadString *s = malloc(sizeof(TriadString));
    s->refcount = 1;
    s->len = len;
    s->cap = len < 16 ? 16 : len + 1;
    s->data = malloc(s->cap);
    int32_t pos = 0;
    if (a && a->len > 0) { memcpy(s->data, a->data, a->len); pos = a->len; }
    if (b && b->len > 0) { memcpy(s->data + pos, b->data, b->len); pos += b->len; }
    s->data[pos] = '\0';
    return s;
}

bool triad_str_eq(TriadString *a, TriadString *b) {
    if (a == b) return true;
    if (!a || !b) return false;
    if (a->len != b->len) return false;
    return memcmp(a->data, b->data, a->len) == 0;
}

int32_t triad_str_hash(TriadString *s) {
    if (!s) return 0;
    int32_t h = 5381;
    for (int32_t i = 0; i < s->len; i++)
        h = ((h << 5) + h) + (unsigned char)s->data[i];
    return h;
}

int32_t triad_str_len(TriadString *s) {
    return s ? s->len : 0;
}

TriadString *triad_str_slice(TriadString *s, int32_t start, int32_t end, int32_t step) {
    if (!s) return triad_str_new("");
    if (start < 0) start += s->len;
    if (end < 0) end += s->len;
    if (start < 0) start = 0;
    if (end > s->len) end = s->len;
    if (step == 0) step = 1;

    if (step > 0) {
        if (start >= end) return triad_str_new("");
        int32_t slen = ((end - start) + step - 1) / step;
        char *buf = malloc(slen + 1);
        int32_t p = 0;
        for (int32_t i = start; i < end; i += step)
            buf[p++] = s->data[i];
        buf[p] = '\0';
        TriadString *r = triad_str_new_len(buf, p);
        free(buf);
        return r;
    } else {
        if (start >= s->len) start = s->len - 1;
        if (end < -1) end = -1;
        if (start <= end) return triad_str_new("");
        int32_t slen = ((start - end) + (-step) - 1) / (-step);
        char *buf = malloc(slen + 1);
        int32_t p = 0;
        for (int32_t i = start; i > end; i += step)
            buf[p++] = s->data[i];
        buf[p] = '\0';
        TriadString *r = triad_str_new_len(buf, p);
        free(buf);
        return r;
    }
}

TriadString *triad_str_upper(TriadString *s) {
    if (!s) return triad_str_new("");
    char *buf = malloc(s->len + 1);
    for (int32_t i = 0; i < s->len; i++)
        buf[i] = (char)toupper((unsigned char)s->data[i]);
    buf[s->len] = '\0';
    TriadString *r = triad_str_new_len(buf, s->len);
    free(buf);
    return r;
}

TriadString *triad_str_lower(TriadString *s) {
    if (!s) return triad_str_new("");
    char *buf = malloc(s->len + 1);
    for (int32_t i = 0; i < s->len; i++)
        buf[i] = (char)tolower((unsigned char)s->data[i]);
    buf[s->len] = '\0';
    TriadString *r = triad_str_new_len(buf, s->len);
    free(buf);
    return r;
}

TriadString *triad_str_strip(TriadString *s) {
    if (!s) return triad_str_new("");
    int32_t start = 0, end = s->len - 1;
    while (start <= end && isspace((unsigned char)s->data[start])) start++;
    while (end >= start && isspace((unsigned char)s->data[end])) end--;
    return triad_str_new_len(s->data + start, end - start + 1);
}

TriadString *triad_str_replace(TriadString *s, const char *old, const char *rep) {
    if (!s || !old || !rep) return triad_str_new("");
    int32_t olen = (int32_t)strlen(old);
    int32_t rlen = (int32_t)strlen(rep);
    if (olen == 0) return triad_str_copy(s);

    int32_t count = 0;
    const char *p = s->data;
    while ((p = strstr(p, old)) != NULL) { count++; p += olen; }

    int32_t newlen = s->len + count * (rlen - olen);
    char *buf = malloc(newlen + 1);
    char *out = buf;
    p = s->data;
    const char *prev = s->data;
    while ((p = strstr(p, old)) != NULL) {
        int32_t chunk = (int32_t)(p - prev);
        memcpy(out, prev, chunk);
        out += chunk;
        memcpy(out, rep, rlen);
        out += rlen;
        p += olen;
        prev = p;
    }
    int32_t remaining = s->len - (int32_t)(prev - s->data);
    memcpy(out, prev, remaining);
    out[remaining] = '\0';

    TriadString *r = triad_str_new_len(buf, (int32_t)(out - buf) + remaining);
    free(buf);
    return r;
}

TriadList *triad_str_split(TriadString *s, const char *sep) {
    TriadList *l = triad_list_new();
    if (!s) return l;
    if (!sep || sep[0] == '\0') {
        triad_list_push(l, (TriadValue){.tag = TRIAD_STRING, .as = {.sval = triad_str_copy(s)}});
        return l;
    }
    int32_t slen = (int32_t)strlen(sep);
    int32_t start = 0;
    for (int32_t i = 0; i <= s->len - slen; ) {
        if (memcmp(s->data + i, sep, slen) == 0) {
            TriadString *part = triad_str_new_len(s->data + start, i - start);
            triad_list_push(l, (TriadValue){.tag = TRIAD_STRING, .as = {.sval = part}});
            i += slen;
            start = i;
        } else {
            i++;
        }
    }
    TriadString *last = triad_str_new_len(s->data + start, s->len - start);
    triad_list_push(l, (TriadValue){.tag = TRIAD_STRING, .as = {.sval = last}});
    return l;
}

bool triad_str_starts_with(TriadString *s, const char *prefix) {
    if (!s || !prefix) return false;
    int32_t plen = (int32_t)strlen(prefix);
    if (plen > s->len) return false;
    return memcmp(s->data, prefix, plen) == 0;
}

bool triad_str_ends_with(TriadString *s, const char *suffix) {
    if (!s || !suffix) return false;
    int32_t slen = (int32_t)strlen(suffix);
    if (slen > s->len) return false;
    return memcmp(s->data + s->len - slen, suffix, slen) == 0;
}

bool triad_str_contains(TriadString *s, const char *sub) {
    if (!s || !sub) return false;
    return strstr(s->data, sub) != NULL;
}

TriadString *triad_str_format(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int32_t len = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    char *buf = malloc(len + 1);
    va_start(ap, fmt);
    vsnprintf(buf, len + 1, fmt, ap);
    va_end(ap);
    TriadString *s = triad_str_new_len(buf, len);
    free(buf);
    return s;
}
