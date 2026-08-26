#ifndef TRIAD_CRYPTO_H
#define TRIAD_CRYPTO_H

#include <stdint.h>
#include <stddef.h>

#define SHA256_BLOCK_SIZE 64
#define SHA256_DIGEST_SIZE 32
#define SHA256_STATE_SIZE 8

typedef struct {
    uint32_t state[SHA256_STATE_SIZE];
    uint64_t count;
    uint8_t buffer[SHA256_BLOCK_SIZE];
} Sha256Ctx;

void triad_sha256_init(Sha256Ctx *ctx);
void triad_sha256_update(Sha256Ctx *ctx, const uint8_t *data, size_t len);
void triad_sha256_final(Sha256Ctx *ctx, uint8_t out[SHA256_DIGEST_SIZE]);

void triad_sha256(const uint8_t *data, size_t len, uint8_t out[32]);

#define AES_BLOCK_SIZE 16
#define AES_KEY_SIZE_128 16
#define AES_KEY_SIZE_192 24
#define AES_KEY_SIZE_256 32
#define AES_MAX_KEY_SIZE 32
#define AES_MAX_ROUNDS 14

typedef struct {
    uint32_t enc_key[60];
    uint32_t dec_key[60];
    int rounds;
    int key_size;
} AesCtx;

int triad_aes_init(AesCtx *ctx, const uint8_t *key, size_t key_size);
void triad_aes_encrypt_block(AesCtx *ctx, const uint8_t in[16], uint8_t out[16]);
void triad_aes_decrypt_block(AesCtx *ctx, const uint8_t in[16], uint8_t out[16]);

void triad_aes_cbc_encrypt(AesCtx *ctx, const uint8_t *in, uint8_t *out, size_t len, const uint8_t iv[16]);
void triad_aes_cbc_decrypt(AesCtx *ctx, const uint8_t *in, uint8_t *out, size_t len, const uint8_t iv[16]);

#define HMAC_BLOCK_SIZE 128
#define HMAC_DIGEST_SIZE 32

void triad_hmac_sha256(const uint8_t *key, size_t key_len,
                       const uint8_t *data, size_t data_len,
                       uint8_t out[32]);

#define RNG_STATE_SIZE 32

typedef struct {
    uint64_t state[4];
    uint64_t buffer[4];
    int initialized;
} TriadRng;

void triad_rng_init(TriadRng *rng);
void triad_rng_reseed(TriadRng *rng, const uint8_t *seed, size_t seed_len);
void triad_rng_bytes(TriadRng *rng, uint8_t *out, size_t len);
uint64_t triad_rng_u64(TriadRng *rng);
uint32_t triad_rng_u32(TriadRng *rng);
double triad_rng_double(TriadRng *rng);

void triad_random_bytes(uint8_t *out, size_t len);
uint64_t triad_random_u64(void);
uint32_t triad_random_u32(void);

void triad_secure_zero(void *ptr, size_t len);
void triad_secure_memcpy(void *dst, const void *src, size_t len);
int triad_ct_memcmp(const void *a, const void *b, size_t len);

#define CRC32_TABLE_SIZE 256
uint32_t triad_crc32(const uint8_t *data, size_t len);
uint32_t triad_crc32_update(uint32_t crc, const uint8_t *data, size_t len);

#endif