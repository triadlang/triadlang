#ifndef TRIAD_KERNEL_H
#define TRIAD_KERNEL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define MAX_PROCS      64
#define MAX_THREADS    256
#define MAX_FDS        32
#define MAX_CAPS       16
#define STACK_SIZE     16384
#define HEAP_SIZE      (32 * 1024 * 1024)
#define TIMER_HZ       100
#define SYSCALL_MAX    64

typedef struct TriadCap TriadCap;
typedef struct TriadThread TriadThread;
typedef struct TriadProc TriadProc;
typedef struct TriadFd TriadFd;
typedef struct TriadSandbox TriadSandbox;

typedef enum {
    CAP_FS_READ   = 1 << 0,
    CAP_FS_WRITE  = 1 << 1,
    CAP_FS_REMOVE = 1 << 2,
    CAP_NET_GET   = 1 << 3,
    CAP_NET_POST  = 1 << 4,
    CAP_PROC_EXEC = 1 << 5,
    CAP_PROC_SPAWN= 1 << 6,
    CAP_ML_LOCAL  = 1 << 7,
    CAP_ML_REMOTE = 1 << 8,
    CAP_PY_NATIVE = 1 << 9,
    CAP_CCALL     = 1 << 10,
    CAP_KERNEL    = 1 << 11,
    CAP_AI_TOOL   = 1 << 12,
    CAP_DEBUG     = 1 << 13,
    CAP_RAW_IO    = 1 << 14,
    CAP_MMAP      = 1 << 15,
    CAP_SIGNAL    = 1 << 16,
    CAP_CAP_MGMT  = 1 << 17,
} TriadCapBits;

struct TriadCap {
    uint32_t bits;
    uint32_t bounding;
    uint32_t ambient;
    char     fs_read_patterns[MAX_CAPS][64];
    char     fs_write_patterns[MAX_CAPS][64];
    int      n_read;
    int      n_write;
    char     net_get_hosts[MAX_CAPS][64];
    char     net_post_hosts[MAX_CAPS][64];
    int      n_get;
    int      n_post;
    char     ml_models[MAX_CAPS][64];
    int      n_models;
};

typedef enum {
    PROC_T0_KERNEL  = 0,
    PROC_T1_SYSTEM   = 1,
    PROC_T2_SIGNED   = 2,
    PROC_T3_USER     = 3,
    PROC_T4_AI       = 4,
} TriadTier;

typedef enum {
    THREAD_READY   = 0,
    THREAD_RUNNING = 1,
    THREAD_BLOCKED  = 2,
    THREAD_DEAD     = 3,
} TriadThreadState;

struct TriadFd {
    bool     used;
    int      inode;
    uint32_t offset;
    uint8_t  mode;
};

typedef struct TriadSavedCtx {
    uint64_t r15, r14, r13, r12, r11, r10, r9, r8;
    uint64_t rdi, rsi, rbp, rdx, rcx, rbx, rax;
    uint64_t rip, rsp, rflags;
} TriadSavedCtx;

struct TriadThread {
    int        id;
    int        proc_id;
    TriadThreadState state;
    TriadSavedCtx ctx;
    uint64_t  *stack_base;
    uint64_t  stack_size;
    uint64_t  quantum_ticks;
    uint64_t  total_ticks;
    void     *triad_closure;
};

struct TriadProc {
    int        id;
    TriadTier  tier;
    TriadCap   caps;
    TriadFd    fds[MAX_FDS];
    TriadSandbox *sandbox;
    char       name[64];
    char       cwd[256];
    int        parent_id;
    bool       alive;
    uint64_t   memory_used;
    uint64_t   stack_canary;
    uint64_t   aslr_seed;
    uint32_t   audit_flags;
    uint32_t   flags;
    uint64_t   create_time;
    uint64_t   last_syscall;
    uint32_t   syscall_count;
};

#endif
