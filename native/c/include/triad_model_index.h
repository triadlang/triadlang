#ifndef TRIAD_MODEL_INDEX_H
#define TRIAD_MODEL_INDEX_H

#include "triad_safetensors.h"

typedef struct {
    char *name;
    char *shard;
} TriadTensorMapEntry;

typedef struct {
    char *name;
    TriadSafeTensorFile file;
} TriadOpenShard;

typedef struct {
    char *dir;
    int32_t nentries;
    TriadTensorMapEntry *entries;
    int32_t nshards;
    TriadOpenShard *shards;
} TriadModelIndex;

int  triad_model_index_load(TriadModelIndex *idx, const char *model_dir);
void triad_model_index_free(TriadModelIndex *idx);

const char *triad_model_index_shard_for(const TriadModelIndex *idx,
                                        const char *tensor_name);

const TriadSafeTensorInfo *triad_model_find_tensor(TriadModelIndex *idx,
                                                   const char *tensor_name,
                                                   TriadSafeTensorFile **out_file);

int triad_model_read_row_f64(TriadModelIndex *idx, const char *tensor_name,
                             int64_t row, double *dst, int64_t dst_len);

#endif
