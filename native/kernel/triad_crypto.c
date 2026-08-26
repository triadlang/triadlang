#include "triad_crypto.h"
#include "triad_kernel.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static const uint32_t sha256_init[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

static const uint32_t sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define EP1(x) (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define SIG0(x) (ROTR32(x, 7) ^ ROTR32(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR32(x, 17) ^ ROTR32(x, 19) ^ ((x) >> 10))

static void sha256_transform(Sha256Ctx *ctx, const uint8_t *block) {
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;
    int i;

    for (i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    }
    for (i = 16; i < 64; i++) {
        w[i] = SIG1(w[i - 2]) + w[i - 7] + SIG0(w[i - 15]) + w[i - 16];
    }

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    for (i = 0; i < 64; i++) {
        t1 = h + EP1(e) + CH(e, f, g) + sha256_k[i] + w[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

void triad_sha256_init(Sha256Ctx *ctx) {
    for (int i = 0; i < 8; i++) {
        ctx->state[i] = sha256_init[i];
    }
    ctx->count = 0;
}

void triad_sha256_update(Sha256Ctx *ctx, const uint8_t *data, size_t len) {
    size_t buffer_idx = (size_t)(ctx->count % SHA256_BLOCK_SIZE);
    ctx->count += len;

    if (buffer_idx + len < SHA256_BLOCK_SIZE) {
        for (size_t i = 0; i < len; i++) {
            ctx->buffer[buffer_idx + i] = data[i];
        }
        return;
    }

    if (buffer_idx > 0) {
        size_t fill = SHA256_BLOCK_SIZE - buffer_idx;
        for (size_t i = 0; i < fill; i++) {
            ctx->buffer[buffer_idx + i] = data[i];
        }
        sha256_transform(ctx, ctx->buffer);
        data += fill;
        len -= fill;
    }

    while (len >= SHA256_BLOCK_SIZE) {
        sha256_transform(ctx, data);
        data += SHA256_BLOCK_SIZE;
        len -= SHA256_BLOCK_SIZE;
    }

    for (size_t i = 0; i < len; i++) {
        ctx->buffer[i] = data[i];
    }
}

void triad_sha256_final(Sha256Ctx *ctx, uint8_t out[32]) {
    uint8_t pad[128];
    uint64_t bit_count = ctx->count * 8;
    size_t pad_len;
    size_t i;

    pad_len = (ctx->count % SHA256_BLOCK_SIZE < 56) ? 56 - (ctx->count % SHA256_BLOCK_SIZE) : 120 - (ctx->count % SHA256_BLOCK_SIZE);

    for (i = 0; i < pad_len; i++) {
        pad[i] = (i == 0) ? 0x80 : 0x00;
    }
    pad[pad_len] = (uint8_t)(bit_count >> 56);
    pad[pad_len + 1] = (uint8_t)(bit_count >> 48);
    pad[pad_len + 2] = (uint8_t)(bit_count >> 40);
    pad[pad_len + 3] = (uint8_t)(bit_count >> 32);
    pad[pad_len + 4] = (uint8_t)(bit_count >> 24);
    pad[pad_len + 5] = (uint8_t)(bit_count >> 16);
    pad[pad_len + 6] = (uint8_t)(bit_count >> 8);
    pad[pad_len + 7] = (uint8_t)(bit_count);

    triad_sha256_update(ctx, pad, pad_len + 8);

    for (i = 0; i < 8; i++) {
        out[i * 4] = (uint8_t)(ctx->state[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        out[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

void triad_sha256(const uint8_t *data, size_t len, uint8_t out[32]) {
    Sha256Ctx ctx;
    triad_sha256_init(&ctx);
    triad_sha256_update(&ctx, data, len);
    triad_sha256_final(&ctx, out);
}

static const uint8_t aes_sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

static const uint8_t aes_inv_sbox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

static const uint8_t aes_rcon[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

#define GET_UINT32(p) (((uint32_t)(p)[0] << 24) | ((uint32_t)(p)[1] << 16) | ((uint32_t)(p)[2] << 8) | ((uint32_t)(p)[3]))
#define PUT_UINT32(n, p) { (p)[0] = (uint8_t)((n) >> 24); (p)[1] = (uint8_t)((n) >> 16); (p)[2] = (uint8_t)((n) >> 8); (p)[3] = (uint8_t)(n); }

int triad_aes_init(AesCtx *ctx, const uint8_t *key, size_t key_size) {
    if (!ctx || !key) return -1;
    if (key_size != 16 && key_size != 24 && key_size != 32) return -1;

    ctx->key_size = (int)key_size;
    ctx->rounds = (key_size == 16) ? 10 : (key_size == 24) ? 12 : 14;

    uint32_t *rk = ctx->enc_key;
    rk[0] = GET_UINT32(key);
    rk[1] = GET_UINT32(key + 4);
    rk[2] = GET_UINT32(key + 8);
    rk[3] = GET_UINT32(key + 12);

    if (key_size > 16) {
        rk[4] = GET_UINT32(key + 16);
        rk[5] = GET_UINT32(key + 20);
    }
    if (key_size > 24) {
        rk[6] = GET_UINT32(key + 24);
        rk[7] = GET_UINT32(key + 28);
    }

    int i;
    for (i = (key_size >> 2); i < (ctx->rounds + 1) * 4; i++) {
        uint32_t temp = rk[i - 1];
        if (!(i & 3)) {
            temp = aes_sbox[(temp >> 16) & 0xFF] << 24 ^
                   aes_sbox[(temp >> 8) & 0xFF] << 16 ^
                   aes_sbox[temp & 0xFF] << 8 ^
                   aes_sbox[(temp >> 24) & 0xFF] ^
                   (uint32_t)aes_rcon[i >> 2] << 24;
        } else if (key_size == 32 && (i & 3) == 1) {
            temp = (uint32_t)aes_sbox[(temp >> 24) & 0xFF] << 24 ^
                   (uint32_t)aes_sbox[(temp >> 16) & 0xFF] << 16 ^
                   (uint32_t)aes_sbox[(temp >> 8) & 0xFF] << 8 ^
                   (uint32_t)aes_sbox[temp & 0xFF];
        }
        rk[i] = rk[i - (key_size >> 2)] ^ temp;
    }

    uint32_t *dk = ctx->dec_key;
    dk[ctx->rounds * 4] = rk[0];
    dk[ctx->rounds * 4 + 1] = rk[1];
    dk[ctx->rounds * 4 + 2] = rk[2];
    dk[ctx->rounds * 4 + 3] = rk[3];

    for (i = ctx->rounds - 1; i >= 1; i--) {
        dk[i * 4] = rk[(ctx->rounds - i) * 4];
        dk[i * 4 + 1] = rk[(ctx->rounds - i) * 4 + 1];
        dk[i * 4 + 2] = rk[(ctx->rounds - i) * 4 + 2];
        dk[i * 4 + 3] = rk[(ctx->rounds - i) * 4 + 3];
    }

    dk[0] = rk[ctx->rounds * 4];
    dk[1] = rk[ctx->rounds * 4 + 1];
    dk[2] = rk[ctx->rounds * 4 + 2];
    dk[3] = rk[ctx->rounds * 4 + 3];

    return 0;
}

#define AES_SUB_WORD(w) ((uint32_t)aes_sbox[(w) >> 24 & 0xFF] << 24 ^ (uint32_t)aes_sbox[(w) >> 16 & 0xFF] << 16 ^ (uint32_t)aes_sbox[(w) >> 8 & 0xFF] << 8 ^ (uint32_t)aes_sbox[(w) & 0xFF])
#define AES_ROT_WORD(w) (((w) << 8) | ((w) >> 24))

static void aes_enc_block(AesCtx *ctx, const uint8_t in[16], uint8_t out[16]) {
    uint32_t s0 = GET_UINT32(in);
    uint32_t s1 = GET_UINT32(in + 4);
    uint32_t s2 = GET_UINT32(in + 8);
    uint32_t s3 = GET_UINT32(in + 12);
    uint32_t *rk = ctx->enc_key;

    s0 ^= rk[0];
    s1 ^= rk[1];
    s2 ^= rk[2];
    s3 ^= rk[3];

    for (int r = 1; r < ctx->rounds; r++) {
        uint32_t t0 = aes_sbox[(s0 >> 24) & 0xFF] << 24 ^ aes_sbox[(s1 >> 16) & 0xFF] << 16 ^ aes_sbox[(s2 >> 8) & 0xFF] << 8 ^ aes_sbox[s3 & 0xFF] ^ rk[r * 4];
        uint32_t t1 = aes_sbox[(s1 >> 24) & 0xFF] << 24 ^ aes_sbox[(s2 >> 16) & 0xFF] << 16 ^ aes_sbox[(s3 >> 8) & 0xFF] << 8 ^ aes_sbox[s0 & 0xFF] ^ rk[r * 4 + 1];
        uint32_t t2 = aes_sbox[(s2 >> 24) & 0xFF] << 24 ^ aes_sbox[(s3 >> 16) & 0xFF] << 16 ^ aes_sbox[(s0 >> 8) & 0xFF] << 8 ^ aes_sbox[s1 & 0xFF] ^ rk[r * 4 + 2];
        uint32_t t3 = aes_sbox[(s3 >> 24) & 0xFF] << 24 ^ aes_sbox[(s0 >> 16) & 0xFF] << 16 ^ aes_sbox[(s1 >> 8) & 0xFF] << 8 ^ aes_sbox[s2 & 0xFF] ^ rk[r * 4 + 3];
        s0 = t0;
        s1 = t1;
        s2 = t2;
        s3 = t3;
    }

    s0 = (aes_sbox[(s0 >> 24) & 0xFF] << 24 | aes_sbox[(s1 >> 16) & 0xFF] << 16 | aes_sbox[(s2 >> 8) & 0xFF] << 8 | aes_sbox[s3 & 0xFF]) ^ rk[ctx->rounds * 4];
    s1 = (aes_sbox[(s1 >> 24) & 0xFF] << 24 | aes_sbox[(s2 >> 16) & 0xFF] << 16 | aes_sbox[(s3 >> 8) & 0xFF] << 8 | aes_sbox[s0 & 0xFF]) ^ rk[ctx->rounds * 4 + 1];
    s2 = (aes_sbox[(s2 >> 24) & 0xFF] << 24 | aes_sbox[(s3 >> 16) & 0xFF] << 16 | aes_sbox[(s0 >> 8) & 0xFF] << 8 | aes_sbox[s1 & 0xFF]) ^ rk[ctx->rounds * 4 + 2];
    s3 = (aes_sbox[(s3 >> 24) & 0xFF] << 24 | aes_sbox[(s0 >> 16) & 0xFF] << 16 | aes_sbox[(s1 >> 8) & 0xFF] << 8 | aes_sbox[s2 & 0xFF]) ^ rk[ctx->rounds * 4 + 3];

    PUT_UINT32(s0, out);
    PUT_UINT32(s1, out + 4);
    PUT_UINT32(s2, out + 8);
    PUT_UINT32(s3, out + 12);
}

static void aes_dec_block(AesCtx *ctx, const uint8_t in[16], uint8_t out[16]) {
    uint32_t s0 = GET_UINT32(in) ^ ctx->enc_key[ctx->rounds * 4];
    uint32_t s1 = GET_UINT32(in + 4) ^ ctx->enc_key[ctx->rounds * 4 + 1];
    uint32_t s2 = GET_UINT32(in + 8) ^ ctx->enc_key[ctx->rounds * 4 + 2];
    uint32_t s3 = GET_UINT32(in + 12) ^ ctx->enc_key[ctx->rounds * 4 + 3];

    for (int r = ctx->rounds - 1; r > 0; r--) {
        uint32_t t0 = aes_inv_sbox[(s0 >> 24) & 0xFF] << 24 ^ aes_inv_sbox[(s3 >> 16) & 0xFF] << 16 ^ aes_inv_sbox[(s2 >> 8) & 0xFF] << 8 ^ aes_inv_sbox[s1 & 0xFF];
        uint32_t t1 = aes_inv_sbox[(s1 >> 24) & 0xFF] << 24 ^ aes_inv_sbox[(s0 >> 16) & 0xFF] << 16 ^ aes_inv_sbox[(s3 >> 8) & 0xFF] << 8 ^ aes_inv_sbox[s2 & 0xFF];
        uint32_t t2 = aes_inv_sbox[(s2 >> 24) & 0xFF] << 24 ^ aes_inv_sbox[(s1 >> 16) & 0xFF] << 16 ^ aes_inv_sbox[(s0 >> 8) & 0xFF] << 8 ^ aes_inv_sbox[s3 & 0xFF];
        uint32_t t3 = aes_inv_sbox[(s3 >> 24) & 0xFF] << 24 ^ aes_inv_sbox[(s2 >> 16) & 0xFF] << 16 ^ aes_inv_sbox[(s1 >> 8) & 0xFF] << 8 ^ aes_inv_sbox[s0 & 0xFF];
        s0 = t0 ^ ctx->enc_key[r * 4];
        s1 = t1 ^ ctx->enc_key[r * 4 + 1];
        s2 = t2 ^ ctx->enc_key[r * 4 + 2];
        s3 = t3 ^ ctx->enc_key[r * 4 + 3];
    }

    s0 = (uint32_t)aes_inv_sbox[(s0 >> 24) & 0xFF] << 24 | (uint32_t)aes_inv_sbox[(s3 >> 16) & 0xFF] << 16 | (uint32_t)aes_inv_sbox[(s2 >> 8) & 0xFF] << 8 | (uint32_t)aes_inv_sbox[s1 & 0xFF] ^ ctx->enc_key[0];
    s1 = (uint32_t)aes_inv_sbox[(s1 >> 24) & 0xFF] << 24 | (uint32_t)aes_inv_sbox[(s0 >> 16) & 0xFF] << 16 | (uint32_t)aes_inv_sbox[(s3 >> 8) & 0xFF] << 8 | (uint32_t)aes_inv_sbox[s2 & 0xFF] ^ ctx->enc_key[1];
    s2 = (uint32_t)aes_inv_sbox[(s2 >> 24) & 0xFF] << 24 | (uint32_t)aes_inv_sbox[(s1 >> 16) & 0xFF] << 16 | (uint32_t)aes_inv_sbox[(s0 >> 8) & 0xFF] << 8 | (uint32_t)aes_inv_sbox[s3 & 0xFF] ^ ctx->enc_key[2];
    s3 = (uint32_t)aes_inv_sbox[(s3 >> 24) & 0xFF] << 24 | (uint32_t)aes_inv_sbox[(s2 >> 16) & 0xFF] << 16 | (uint32_t)aes_inv_sbox[(s1 >> 8) & 0xFF] << 8 | (uint32_t)aes_inv_sbox[s0 & 0xFF] ^ ctx->enc_key[3];

    PUT_UINT32(s0, out);
    PUT_UINT32(s1, out + 4);
    PUT_UINT32(s2, out + 8);
    PUT_UINT32(s3, out + 12);
}

void triad_aes_encrypt_block(AesCtx *ctx, const uint8_t in[16], uint8_t out[16]) {
    aes_enc_block(ctx, in, out);
}

void triad_aes_decrypt_block(AesCtx *ctx, const uint8_t in[16], uint8_t out[16]) {
    aes_dec_block(ctx, in, out);
}

void triad_aes_cbc_encrypt(AesCtx *ctx, const uint8_t *in, uint8_t *out, size_t len, const uint8_t iv[16]) {
    uint8_t block[16];
    uint8_t last[16];
    for (int i = 0; i < 16; i++) last[i] = iv[i];

    for (size_t i = 0; i < len; i += 16) {
        for (int j = 0; j < 16 && i + j < len; j++) {
            block[j] = in[i + j] ^ last[j];
        }
        aes_enc_block(ctx, block, out + i);
        for (int j = 0; j < 16; j++) last[j] = out[i + j];
    }
}

void triad_aes_cbc_decrypt(AesCtx *ctx, const uint8_t *in, uint8_t *out, size_t len, const uint8_t iv[16]) {
    uint8_t last[16];
    for (int i = 0; i < 16; i++) last[i] = iv[i];

    for (size_t i = 0; i < len; i += 16) {
        uint8_t block[16];
        aes_dec_block(ctx, in + i, block);
        for (int j = 0; j < 16 && i + j < len; j++) {
            out[i + j] = block[j] ^ last[j];
            last[j] = in[i + j];
        }
    }
}

void triad_hmac_sha256(const uint8_t *key, size_t key_len, const uint8_t *data, size_t data_len, uint8_t out[32]) {
    uint8_t k_ipad[HMAC_BLOCK_SIZE];
    uint8_t k_opad[HMAC_BLOCK_SIZE];
    uint8_t tk[32];
    Sha256Ctx ctx;

    if (key_len > HMAC_BLOCK_SIZE) {
        triad_sha256(key, key_len, tk);
        key = tk;
        key_len = 32;
    }

    for (size_t i = 0; i < HMAC_BLOCK_SIZE; i++) {
        k_ipad[i] = (i < key_len ? key[i] : 0) ^ 0x36;
        k_opad[i] = (i < key_len ? key[i] : 0) ^ 0x5c;
    }

    triad_sha256_init(&ctx);
    triad_sha256_update(&ctx, k_ipad, HMAC_BLOCK_SIZE);
    triad_sha256_update(&ctx, data, data_len);
    triad_sha256_final(&ctx, out);

    triad_sha256_init(&ctx);
    triad_sha256_update(&ctx, k_opad, HMAC_BLOCK_SIZE);
    triad_sha256_update(&ctx, out, 32);
    triad_sha256_final(&ctx, out);
}

static TriadRng global_rng = {0};
static int global_rng_init = 0;

static uint64_t read_tsc_entropy(void) {
    uint64_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return (hi << 32) | lo;
}

static int rdrand_available(void) {
    uint32_t eax, ebx, ecx, edx;
    eax = 1;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(eax));
    return (ecx & (1 << 30)) != 0;
}

static int rdrand64(uint64_t *val) {
    int retries = 10;
    while (retries-- > 0) {
        unsigned char ok = 0;
        __asm__ volatile ("rdrand %0; setc %1" : "=r"(*val), "=rm"(ok) :: "cc");
        if (ok) return 1;
    }
    return 0;
}

void triad_rng_init(TriadRng *rng) {
    rng->state[0] = read_tsc_entropy() ^ 0xDEADBEEFCAFEBABEULL;
    rng->state[1] = read_tsc_entropy() ^ 0x123456789ABCDEFULL;
    rng->state[2] = (uint64_t)rng ^ read_tsc_entropy();
    rng->state[3] = read_tsc_entropy() ^ (read_tsc_entropy() << 32);

    if (rdrand_available()) {
        uint64_t rd;
        if (rdrand64(&rd)) rng->state[0] ^= rd;
        if (rdrand64(&rd)) rng->state[1] ^= rd;
    }

    rng->initialized = 1;
}

void triad_rng_reseed(TriadRng *rng, const uint8_t *seed, size_t seed_len) {
    Sha256Ctx ctx;
    uint8_t hash[32];

    triad_sha256_init(&ctx);
    triad_sha256_update(&ctx, (const uint8_t *)rng->state, 32);
    triad_sha256_update(&ctx, seed, seed_len);
    triad_sha256_final(&ctx, hash);

    for (int i = 0; i < 4; i++) {
        rng->state[i] = ((uint64_t)hash[i * 8] << 56) | ((uint64_t)hash[i * 8 + 1] << 48) |
                        ((uint64_t)hash[i * 8 + 2] << 40) | ((uint64_t)hash[i * 8 + 3] << 32) |
                        ((uint64_t)hash[i * 8 + 4] << 24) | ((uint64_t)hash[i * 8 + 5] << 16) |
                        ((uint64_t)hash[i * 8 + 6] << 8) | (uint64_t)hash[i * 8 + 7];
    }
    rng->initialized = 1;
}

void triad_rng_bytes(TriadRng *rng, uint8_t *out, size_t len) {
    if (!rng || !out) return;

    uint64_t s0 = rng->state[0];
    uint64_t s1 = rng->state[1];
    uint64_t s2 = rng->state[2];
    uint64_t s3 = rng->state[3];

    for (size_t i = 0; i < len; i += 8) {
        s0 = s0 ^ (s0 << 23);
        s1 = s1 ^ (s1 >> 17);
        s2 = s2 ^ (s2 << 19);
        s3 = s3 ^ (s3 >> 27);

        uint64_t t = s0 ^ s1 ^ s2 ^ s3;
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = t;

        for (size_t j = 0; j < 8 && i + j < len; j++) {
            out[i + j] = (uint8_t)(t >> (j * 8));
        }
    }

    rng->state[0] = s0;
    rng->state[1] = s1;
    rng->state[2] = s2;
    rng->state[3] = s3;
}

uint64_t triad_rng_u64(TriadRng *rng) {
    uint64_t val;
    triad_rng_bytes(rng, (uint8_t *)&val, sizeof(val));
    return val;
}

uint32_t triad_rng_u32(TriadRng *rng) {
    uint32_t val;
    triad_rng_bytes(rng, (uint8_t *)&val, sizeof(val));
    return val;
}

double triad_rng_double(TriadRng *rng) {
    uint64_t val = triad_rng_u64(rng);
    return (double)(val >> 11) / (double)(1ULL << 53);
}

void triad_random_bytes(uint8_t *out, size_t len) {
    if (!global_rng_init) {
        triad_rng_init(&global_rng);
        global_rng_init = 1;
    }
    triad_rng_bytes(&global_rng, out, len);
}

uint64_t triad_random_u64(void) {
    uint64_t val;
    triad_random_bytes((uint8_t *)&val, sizeof(val));
    return val;
}

uint32_t triad_random_u32(void) {
    uint32_t val;
    triad_random_bytes((uint8_t *)&val, sizeof(val));
    return val;
}

void triad_secure_memcpy(void *dst, const void *src, size_t len) {
    volatile uint8_t *d = (volatile uint8_t *)dst;
    const volatile uint8_t *s = (const volatile uint8_t *)src;
    for (size_t i = 0; i < len; i++) {
        d[i] = s[i];
    }
    __asm__ volatile ("" : : "r"(d), "r"(s) : "memory");
}

int triad_ct_memcmp(const void *a, const void *b, size_t len) {
    const volatile uint8_t *aa = (const volatile uint8_t *)a;
    const volatile uint8_t *bb = (const volatile uint8_t *)b;
    volatile uint8_t result = 0;
    for (size_t i = 0; i < len; i++) {
        result |= aa[i] ^ bb[i];
    }
    return result != 0;
}

static uint32_t crc32_table[CRC32_TABLE_SIZE];
static int crc32_table_init = 0;

static void init_crc32_table(void) {
    for (uint32_t i = 0; i < CRC32_TABLE_SIZE; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1)
                crc = (crc >> 1) ^ 0xEDB88320;
            else
                crc >>= 1;
        }
        crc32_table[i] = crc;
    }
    crc32_table_init = 1;
}

uint32_t triad_crc32(const uint8_t *data, size_t len) {
    return triad_crc32_update(0xFFFFFFFF, data, len) ^ 0xFFFFFFFF;
}

uint32_t triad_crc32_update(uint32_t crc, const uint8_t *data, size_t len) {
    if (!crc32_table_init) init_crc32_table();

    for (size_t i = 0; i < len; i++) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc;
}