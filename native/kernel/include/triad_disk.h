#ifndef TRIAD_DISK_H
#define TRIAD_DISK_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define DISK_SECTOR_SIZE    512
#define DISK_CACHE_SIZE     16
#define DISKFS_MAGIC        0x54524941
#define DISKFS_VERSION      1

typedef struct DiskFsSuperblock {
    uint32_t magic;
    uint32_t version;
    uint64_t total_blocks;
    uint64_t free_blocks;
    uint64_t inode_count;
    uint64_t root_inode;
    uint64_t bitmap_start;
    uint64_t bitmap_size;
    char     label[64];
} DiskFsSuperblock;

typedef struct DiskFsInode {
    uint64_t ino;
    uint64_t parent_ino;
    uint32_t type;
    uint32_t mode;
    uint64_t size;
    uint64_t created;
    uint64_t modified;
    uint64_t first_block;
    char     name[64];
} DiskFsInode;

typedef struct DiskFsDirEntry {
    uint64_t ino;
    char     name[64];
    uint8_t  type;
    uint8_t  padding[7];
} DiskFsDirEntry;

#define DISKFS_TYPE_FILE    1
#define DISKFS_TYPE_DIR     2

int triad_disk_init(void);
int triad_disk_format(uint64_t start_sector, uint64_t num_sectors);
int triad_disk_mount(uint64_t start_sector);
int triad_disk_read_block(uint64_t block, void *buf);
int triad_disk_write_block(uint64_t block, const void *buf);

uint64_t triad_disk_alloc_block(void);
void triad_disk_free_block(uint64_t block);

uint64_t triad_disk_create_file(const char *path);
uint64_t triad_disk_create_dir(const char *path);
uint64_t triad_disk_lookup(const char *path);
int triad_disk_read_file(uint64_t ino, void *buf, uint64_t offset, uint64_t size);
int triad_disk_write_file(uint64_t ino, const void *buf, uint64_t offset, uint64_t size);
int triad_disk_list_dir(uint64_t ino, DiskFsDirEntry *entries, int max);
int triad_disk_remove(uint64_t ino);

void triad_disk_sync(void);
int triad_disk_sync_vfs(void);
int triad_disk_load_vfs(void);

#endif