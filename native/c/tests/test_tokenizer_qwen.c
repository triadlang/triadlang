#include "triad_llm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(const char *label, int ok) {
    printf("  %-32s %s\n", label, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
}

static void join_path(char *out, size_t n, const char *dir, const char *file) {
    size_t len = strlen(dir);
    snprintf(out, n, "%s%s%s", dir, (len > 0 && dir[len - 1] == '/') ? "" : "/", file);
}

static void check_roundtrip(TriadTokenizer *tok, const char *text) {
    int32_t ids[128];
    int32_t n = triad_tokenizer_encode(tok, text, ids, 128);
    int ids_ok = n > 0;
    for (int32_t i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= tok->vocab_size) ids_ok = 0;
    }
    char *decoded = triad_tokenizer_decode(tok, ids, n);
    int decoded_ok = decoded && strstr(decoded, text) != NULL;
    char label[128];
    snprintf(label, sizeof(label), "roundtrip '%s'", text);
    check(label, ids_ok && decoded_ok);
    free(decoded);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "models";
    char vocab_path[4096];
    char merges_path[4096];
    char config_path[4096];
    join_path(vocab_path, sizeof(vocab_path), dir, "vocab.json");
    join_path(merges_path, sizeof(merges_path), dir, "merges.txt");
    join_path(config_path, sizeof(config_path), dir, "tokenizer_config.json");

    printf("=== qwen tokenizer ===\n");

    TriadTokenizer *tok = triad_tokenizer_new();
    if (!tok) {
        fprintf(stderr, "failed to allocate tokenizer\n");
        return 1;
    }
    check("load vocab", triad_tokenizer_load_vocab(tok, vocab_path) == 0);
    check("load merges", triad_tokenizer_load_merges(tok, merges_path) == 0);
    check("load added tokens", triad_tokenizer_load_added_tokens(tok, config_path) == 0);
    check("vocab size", tok->vocab_size >= 248047);

    check("endoftext id", strcmp(triad_tokenizer_id_to_str(tok, 248044),
                                 "<|endoftext|>") == 0);
    check("im_start id", strcmp(triad_tokenizer_id_to_str(tok, 248045),
                                "<|im_start|>") == 0);
    check("im_end id", strcmp(triad_tokenizer_id_to_str(tok, 248046),
                              "<|im_end|>") == 0);

    check_roundtrip(tok, "oi");
    check_roundtrip(tok, "hello");
    check_roundtrip(tok, "TriadLang");

    triad_tokenizer_free(tok);
    printf("qwen tokenizer: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
