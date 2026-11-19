#include "triad_model_index.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static char *mi_strndup(const char *s, size_t n) {
    char *p = malloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = 0;
    return p;
}

static char *mi_strdup(const char *s) {
    return mi_strndup(s, strlen(s));
}

static void join_path(char *out, size_t n, const char *dir, const char *file) {
    size_t len = strlen(dir);
    snprintf(out, n, "%s%s%s", dir, (len > 0 && dir[len - 1] == '/') ? "" : "/", file);
}

static char *parse_json_string(const char **pp) {
    const char *p = *pp;
    if (*p != '"') return NULL;
    p++;
    char *out = malloc(strlen(p) + 1);
    if (!out) return NULL;
    size_t n = 0;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            if (*p) out[n++] = *p++;
        } else {
            out[n++] = *p++;
        }
    }
    out[n] = 0;
    if (*p == '"') p++;
    *pp = p;
    return out;
}

static int add_entry(TriadModelIndex *idx, char *name, char *shard) {
    TriadTensorMapEntry *next = realloc(
        idx->entries, (size_t)(idx->nentries + 1) * sizeof(TriadTensorMapEntry));
    if (!next) return -1;
    idx->entries = next;
    idx->entries[idx->nentries].name = name;
    idx->entries[idx->nentries].shard = shard;
    idx->nentries++;
    return 0;
}

static char *read_file(const char *path, long *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(fp);
        return NULL;
    }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    buf[sz] = 0;
    if (out_size) *out_size = sz;
    return buf;
}

int triad_model_index_load(TriadModelIndex *idx, const char *model_dir) {
    memset(idx, 0, sizeof(*idx));
    idx->dir = mi_strdup(model_dir);
    if (!idx->dir) return -1;

    char path[4096];
    join_path(path, sizeof(path), model_dir, "model.safetensors.index.json");
    long sz = 0;
    char *json = read_file(path, &sz);
    if (!json) {
        triad_model_index_free(idx);
        return -1;
    }

    const char *wm = strstr(json, "\"weight_map\"");
    if (!wm) {
        free(json);
        triad_model_index_free(idx);
        return -1;
    }
    const char *p = strchr(wm, '{');
    if (!p) {
        free(json);
        triad_model_index_free(idx);
        return -1;
    }
    p++;

    while (*p) {
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',')) p++;
        if (*p == '}') break;
        if (*p != '"') break;
        char *name = parse_json_string(&p);
        if (!name) break;
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
        if (*p != ':') {
            free(name);
            break;
        }
        p++;
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
        char *shard = parse_json_string(&p);
        if (!shard) {
            free(name);
            break;
        }
        if (add_entry(idx, name, shard) != 0) {
            free(name);
            free(shard);
            free(json);
            triad_model_index_free(idx);
            return -1;
        }
    }

    free(json);
    return idx->nentries > 0 ? 0 : -1;
}

void triad_model_index_free(TriadModelIndex *idx) {
    if (!idx) return;
    for (int32_t i = 0; i < idx->nentries; i++) {
        free(idx->entries[i].name);
        free(idx->entries[i].shard);
    }
    free(idx->entries);
    for (int32_t i = 0; i < idx->nshards; i++) {
        free(idx->shards[i].name);
        triad_safetensors_close(&idx->shards[i].file);
    }
    free(idx->shards);
    free(idx->dir);
    memset(idx, 0, sizeof(*idx));
}

const char *triad_model_index_shard_for(const TriadModelIndex *idx,
                                        const char *tensor_name) {
    if (!idx || !tensor_name) return NULL;
    for (int32_t i = 0; i < idx->nentries; i++) {
        if (strcmp(idx->entries[i].name, tensor_name) == 0)
            return idx->entries[i].shard;
    }
    return NULL;
}

static TriadSafeTensorFile *open_shard(TriadModelIndex *idx, const char *shard) {
    for (int32_t i = 0; i < idx->nshards; i++)
        if (strcmp(idx->shards[i].name, shard) == 0) return &idx->shards[i].file;

    TriadOpenShard *next = realloc(idx->shards, (size_t)(idx->nshards + 1) * sizeof(TriadOpenShard));
    if (!next) return NULL;
    idx->shards = next;
    TriadOpenShard *slot = &idx->shards[idx->nshards];
    memset(slot, 0, sizeof(*slot));
    slot->name = mi_strdup(shard);
    if (!slot->name) return NULL;

    char path[4096];
    join_path(path, sizeof(path), idx->dir, shard);
    if (triad_safetensors_open(path, &slot->file) != 0) {
        free(slot->name);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }
    idx->nshards++;
    return &slot->file;
}

const TriadSafeTensorInfo *triad_model_find_tensor(TriadModelIndex *idx,
                                                   const char *tensor_name,
                                                   TriadSafeTensorFile **out_file) {
    const char *shard = triad_model_index_shard_for(idx, tensor_name);
    if (!shard) return NULL;
    TriadSafeTensorFile *file = open_shard(idx, shard);
    if (!file) return NULL;
    const TriadSafeTensorInfo *t = triad_safetensors_find(file, tensor_name);
    if (!t) return NULL;
    if (out_file) *out_file = file;
    return t;
}

int triad_model_read_row_f64(TriadModelIndex *idx, const char *tensor_name,
                             int64_t row, double *dst, int64_t dst_len) {
    TriadSafeTensorFile *file = NULL;
    const TriadSafeTensorInfo *t = triad_model_find_tensor(idx, tensor_name, &file);
    if (!t || !file) return -1;
    return triad_safetensors_read_row_f64(file, t, row, dst, dst_len);
}
