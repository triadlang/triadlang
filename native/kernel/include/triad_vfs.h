#ifndef TRIAD_VFS_H
#define TRIAD_VFS_H

#include "triad_kernel.h"

#define VFS_MAX_FILES   256
#define VFS_MAX_PATH    256
#define VFS_BLOCK_SIZE  4096

typedef enum {
    VFS_FILE    = 0,
    VFS_DIR     = 1,
    VFS_DEVICE  = 2,
    VFS_PIPE    = 3,
    VFS_SYMLINK = 4,
} TriadVfsType;

typedef struct {
    bool          used;
    TriadVfsType  type;
    char          name[VFS_MAX_PATH];
    uint32_t      inode;
    uint32_t      parent_inode;
    uint32_t      size;
    uint8_t      *data;
    uint32_t      cap;
    uint32_t      n_children;
    uint32_t      child_inodes[VFS_MAX_FILES];
    uint32_t      owner_proc;
    uint32_t      mode;
    bool          readonly;
} TriadVfsNode;

void triad_vfs_init(void);
int  triad_vfs_mount(const char *target, TriadVfsType type);
int  triad_vfs_mkdir(const char *path);
int  triad_vfs_create(const char *path);
int  triad_vfs_open(const char *path, uint8_t mode);
int  triad_vfs_close(int fd);
int  triad_vfs_read(int fd, void *buf, uint32_t n);
int  triad_vfs_write(int fd, const void *buf, uint32_t n);
int  triad_vfs_seek(int fd, uint32_t offset);
int  triad_vfs_stat(const char *path, TriadVfsNode *out);
int  triad_vfs_listdir(const char *path, char out[][VFS_MAX_PATH], int max);
int  triad_vfs_remove(const char *path);
int  triad_vfs_rename(const char *old_path, const char *new_path);
int  triad_vfs_resolve(TriadSandbox *sb, const char *path, char *out, uint32_t out_len);

#endif