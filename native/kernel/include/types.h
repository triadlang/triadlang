#ifndef TRIAD_KERNEL_TYPES_H
#define TRIAD_KERNEL_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef size_t usize;
typedef intptr_t isize;

typedef struct {
    u64 base;
    u64 limit;
    u32 type;
    u32 flags;
} MemoryRegion;

typedef struct {
    u64 cr3;
    u64 rsp0;
    u64 rsp3;
    u64 rip;
    u64 rflags;
    u16 cs;
    u16 ds;
    u16 ss;
    u16 fs;
    u16 gs;
} CPUState;

typedef struct {
    u64 pid;
    u64 ppid;
    char name[64];
    u64 entry;
    u64 stack;
    u64 heap;
    u64 heap_size;
    CPUState state;
    u8 priority;
    u8 state_flags;
} Process;

typedef struct {
    char name[256];
    u64 size;
    u64 inode;
    u32 mode;
    u32 uid;
    u32 gid;
    u64 atime;
    u64 mtime;
    u64 ctime;
} FileInfo;

typedef struct {
    u64 fd;
    u64 offset;
    u64 size;
    u8 mode;
    u8 flags;
} FileHandle;

#define KERNEL_CS 0x08
#define KERNEL_DS 0x10
#define USER_CS   0x18
#define USER_DS   0x20
#define TSS_SEG   0x28

#endif