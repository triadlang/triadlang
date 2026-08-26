#ifndef TRIAD_SYSCALL_H
#define TRIAD_SYSCALL_H

#include "triad_kernel.h"

typedef enum {
    SYS_EXIT       = 0,
    SYS_WRITE      = 1,
    SYS_READ       = 2,
    SYS_OPEN       = 3,
    SYS_CLOSE      = 4,
    SYS_SEEK       = 5,
    SYS_STAT       = 6,
    SYS_MKDIR      = 7,
    SYS_LISTDIR    = 8,
    SYS_REMOVE     = 9,
    SYS_SPAWN      = 10,
    SYS_WAIT       = 11,
    SYS_GETPID     = 12,
    SYS_GETTICKS   = 13,
    SYS_SLEEP      = 14,
    SYS_CAP_GRANT  = 15,
    SYS_CAP_CHECK  = 16,
    SYS_AI_QUERY   = 17,
    SYS_AI_RECORD  = 18,
    SYS_SOLVER     = 19,
    SYS_NET_GET    = 20,
    SYS_NET_POST   = 21,
    SYS_ML_LOAD    = 22,
    SYS_ML_FORWARD = 23,
    SYS_EQUILIBRIUM= 24,
    SYS_YIELD      = 25,
    SYS_PIPE       = 26,
    SYS_DUP        = 27,
    SYS_BRK        = 28,
    SYS_MMAP       = 29,
    SYS_GETENV     = 30,
    SYS_SETENV     = 31,
    SYS_DISK_READ  = 32,
    SYS_DISK_WRITE = 33,
    SYS_DISK_SYNC  = 34,
    SYS_DISK_MOUNT = 35,
    SYS_NET_SOCKET = 36,
    SYS_NET_CONNECT= 37,
    SYS_NET_LISTEN = 38,
    SYS_NET_ACCEPT = 39,
    SYS_NET_SEND   = 40,
    SYS_NET_RECV   = 41,
    SYS_AI_TOOL_USE= 42,
} TriadSyscall;

int64_t triad_syscall_dispatch(TriadSyscall num,
                               uint64_t a, uint64_t b, uint64_t c,
                               uint64_t d, uint64_t e);

int  triad_syscall_init(void);
void triad_syscall_test(void);

#endif