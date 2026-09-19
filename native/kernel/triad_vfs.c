#include "triad_kernel.h"
#include "triad_mm.h"
#include "triad_vfs.h"
#include "triad_sandbox.h"
#include "triad_crypto.h"

static TriadVfsNode vfs_nodes[VFS_MAX_FILES];
static uint32_t vfs_next_inode = 1;
static uint32_t fd_offset[VFS_MAX_FILES];

static int vfs_find_exact(const char *path) {
    if (!path) return -1;
    for (int i = 0; i < VFS_MAX_FILES; i++) {
        if (vfs_nodes[i].used) {
            int k = 0;
            int j = 0;
            while (path[k] && vfs_nodes[i].name[j] && path[k] == vfs_nodes[i].name[j]) {
                k++; j++;
            }
            if (path[k] == 0 && vfs_nodes[i].name[j] == 0) return i;
        }
    }
    return -1;
}

static int vfs_alloc_node(void) {
    for (int i = 0; i < VFS_MAX_FILES; i++) {
        if (!vfs_nodes[i].used) {
            triad_mm_free(vfs_nodes[i].data);
            vfs_nodes[i].used = false;
            vfs_nodes[i].inode = vfs_next_inode++;
            vfs_nodes[i].data = NULL;
            vfs_nodes[i].cap = 0;
            vfs_nodes[i].size = 0;
            fd_offset[i] = 0;
            return i;
        }
    }
    return -1;
}

static int str_eq(const char *a, const char *b) {
    if (!a || !b) return 0;
    int k = 0;
    while (a[k] && b[k]) {
        if (a[k] != b[k]) return 0;
        k++;
    }
    return a[k] == 0 && b[k] == 0;
}

static int str_starts(const char *s, const char *prefix) {
    if (!s || !prefix) return 0;
    int k = 0;
    while (prefix[k]) {
        if (s[k] != prefix[k]) return 0;
        k++;
    }
    return 1;
}

void triad_vfs_init(void) {
    for (int i = 0; i < VFS_MAX_FILES; i++) {
        vfs_nodes[i].used = false;
        vfs_nodes[i].data = NULL;
        vfs_nodes[i].cap = 0;
        vfs_nodes[i].n_children = 0;
        fd_offset[i] = 0;
    }
    vfs_next_inode = 1;
    int root = vfs_alloc_node();
    vfs_nodes[root].used = true;
    vfs_nodes[root].type = VFS_DIR;
    vfs_nodes[root].name[0] = '/';
    vfs_nodes[root].name[1] = 0;
    vfs_nodes[root].inode = vfs_next_inode++;
    vfs_nodes[root].parent_inode = 0;
    vfs_nodes[root].size = 0;
    vfs_nodes[root].readonly = false;
}

int triad_vfs_mount(const char *target, TriadVfsType type) {
    int idx = vfs_alloc_node();
    if (idx < 0) return -1;
    vfs_nodes[idx].used = true;
    vfs_nodes[idx].type = type;
    int k = 0;
    while (target[k] && k < VFS_MAX_PATH - 1) {
        vfs_nodes[idx].name[k] = target[k];
        k++;
    }
    vfs_nodes[idx].name[k] = 0;
    vfs_nodes[idx].inode = vfs_next_inode++;
    vfs_nodes[idx].parent_inode = 1;
    vfs_nodes[idx].size = 0;
    vfs_nodes[idx].readonly = false;
    return idx;
}

int triad_vfs_mkdir(const char *path) {
    if (vfs_find_exact(path) >= 0) return -1;
    int idx = vfs_alloc_node();
    if (idx < 0) return -1;
    vfs_nodes[idx].used = true;
    vfs_nodes[idx].type = VFS_DIR;
    int k = 0;
    while (path[k] && k < VFS_MAX_PATH - 1) {
        vfs_nodes[idx].name[k] = path[k];
        k++;
    }
    vfs_nodes[idx].name[k] = 0;
    vfs_nodes[idx].inode = vfs_next_inode++;
    vfs_nodes[idx].size = 0;
    vfs_nodes[idx].readonly = false;
    return 0;
}

int triad_vfs_create(const char *path) {
    if (vfs_find_exact(path) >= 0) return -1;
    int idx = vfs_alloc_node();
    if (idx < 0) return -1;
    vfs_nodes[idx].used = true;
    vfs_nodes[idx].type = VFS_FILE;
    int k = 0;
    while (path[k] && k < VFS_MAX_PATH - 1) {
        vfs_nodes[idx].name[k] = path[k];
        k++;
    }
    vfs_nodes[idx].name[k] = 0;
    vfs_nodes[idx].inode = vfs_next_inode++;
    vfs_nodes[idx].size = 0;
    vfs_nodes[idx].cap = VFS_BLOCK_SIZE;
    vfs_nodes[idx].data = (uint8_t *)triad_mm_alloc(VFS_BLOCK_SIZE);
    vfs_nodes[idx].readonly = false;
    return 0;
}

int triad_vfs_open(const char *path, uint8_t mode) {
    int idx = vfs_find_exact(path);
    if (idx < 0) {
        if (mode & 1) {
            if (triad_vfs_create(path) < 0) return -1;
            idx = vfs_find_exact(path);
            if (idx < 0) return -1;
        } else {
            return -1;
        }
    }
    fd_offset[idx] = 0;
    return idx;
}

int triad_vfs_close(int fd) {
    (void)fd;
    return 0;
}

static int is_dev_random(int fd) {
    if (fd < 0 || fd >= VFS_MAX_FILES) return 0;
    if (!vfs_nodes[fd].used) return 0;
    const char *name = vfs_nodes[fd].name;
    return str_eq(name, "/dev/random") || str_eq(name, "/dev/urandom");
}

int triad_vfs_read(int fd, void *buf, uint32_t n) {
    if (fd < 0 || fd >= VFS_MAX_FILES) return -1;
    if (!vfs_nodes[fd].used) return -1;

    if (is_dev_random(fd)) {
        triad_random_bytes((uint8_t *)buf, n);
        return (int)n;
    }

    uint32_t off = fd_offset[fd];
    if (off >= vfs_nodes[fd].size) return 0;
    if (off + n > vfs_nodes[fd].size) {
        n = vfs_nodes[fd].size - off;
    }
    for (uint32_t i = 0; i < n; i++) {
        ((uint8_t *)buf)[i] = vfs_nodes[fd].data[off + i];
    }
    fd_offset[fd] = off + n;
    return (int)n;
}

int triad_vfs_write(int fd, const void *buf, uint32_t n) {
    if (fd < 0 || fd >= VFS_MAX_FILES) return -1;
    if (!vfs_nodes[fd].used) return -1;
    if (vfs_nodes[fd].readonly) return -1;
    uint32_t off = fd_offset[fd];
    if (vfs_nodes[fd].data == NULL) {
        vfs_nodes[fd].data = (uint8_t *)triad_mm_alloc(VFS_BLOCK_SIZE);
        vfs_nodes[fd].cap = VFS_BLOCK_SIZE;
    }
    uint32_t needed = off + n;
    if (needed > vfs_nodes[fd].cap) {
        uint32_t new_cap = vfs_nodes[fd].cap;
        while (new_cap < needed) new_cap *= 2;
        uint8_t *new_data = (uint8_t *)triad_mm_alloc(new_cap);
        if (new_data == NULL) return -1;
        for (uint32_t i = 0; i < vfs_nodes[fd].size; i++) {
            new_data[i] = vfs_nodes[fd].data[i];
        }
        vfs_nodes[fd].data = new_data;
        vfs_nodes[fd].cap = new_cap;
    }
    for (uint32_t i = 0; i < n; i++) {
        vfs_nodes[fd].data[off + i] = ((const uint8_t *)buf)[i];
    }
    if (off + n > vfs_nodes[fd].size) {
        vfs_nodes[fd].size = off + n;
    }
    fd_offset[fd] = off + n;
    return (int)n;
}

int triad_vfs_seek(int fd, uint32_t offset) {
    if (fd < 0 || fd >= VFS_MAX_FILES) return -1;
    if (!vfs_nodes[fd].used) return -1;
    fd_offset[fd] = offset;
    return 0;
}

int triad_vfs_stat(const char *path, TriadVfsNode *out) {
    int idx = vfs_find_exact(path);
    if (idx < 0) return -1;
    *out = vfs_nodes[idx];
    return 0;
}

static int path_parent_match(const char *name, const char *parent) {
    if (parent[0] == '/' && parent[1] == 0) {
        if (name[0] == '/' && name[1] != 0) {
            return 1;
        }
        return 0;
    }
    if (!str_starts(name, parent)) return 0;
    int plen = 0;
    while (parent[plen]) plen++;
    if (name[plen] == '/') return 1;
    return 0;
}

int triad_vfs_listdir(const char *path, char out[][VFS_MAX_PATH], int max) {
    int count = 0;
    int plen = 0;
    while (path[plen]) plen++;

    for (int i = 0; i < VFS_MAX_FILES && count < max; i++) {
        if (!vfs_nodes[i].used) continue;
        if (i == 0) continue;

        if (!path_parent_match(vfs_nodes[i].name, path)) continue;

        int start = plen;
        if (path[plen - 1] == '/') {
            start = plen;
        } else if (vfs_nodes[i].name[plen] == '/') {
            start = plen + 1;
        } else {
            continue;
        }

        int j = 0;
        int k = start;
        while (vfs_nodes[i].name[k] && vfs_nodes[i].name[k] != '/') {
            out[count][j] = vfs_nodes[i].name[k];
            j++; k++;
        }
        out[count][j] = 0;
        count++;
    }
    return count;
}

int triad_vfs_remove(const char *path) {
    int idx = vfs_find_exact(path);
    if (idx < 0) return -1;
    if (vfs_nodes[idx].readonly) return -1;
    triad_mm_free(vfs_nodes[idx].data);
    vfs_nodes[idx].used = false;
    vfs_nodes[idx].data = NULL;
    return 0;
}

int triad_vfs_rename(const char *old_path, const char *new_path) {
    int idx = vfs_find_exact(old_path);
    if (idx < 0) return -1;
    int k = 0;
    while (new_path[k] && k < VFS_MAX_PATH - 1) {
        vfs_nodes[idx].name[k] = new_path[k];
        k++;
    }
    vfs_nodes[idx].name[k] = 0;
    return 0;
}

int triad_vfs_resolve(TriadSandbox *sb, const char *path, char *out, uint32_t out_len) {
    if (path[0] == '/') {
        int k = 0;
        while (path[k] && k < (int)out_len - 1) {
            out[k] = path[k];
            k++;
        }
        out[k] = 0;
    } else {
        int k = 0;
        while (sb->root[k] && k < (int)out_len - 1) {
            out[k] = sb->root[k];
            k++;
        }
        if (k > 0 && sb->root[k - 1] != '/' && k < (int)out_len - 1) {
            out[k] = '/';
            k++;
        }
        int j = 0;
        while (path[j] && k < (int)out_len - 1) {
            out[k] = path[j];
            k++; j++;
        }
        out[k] = 0;
    }
    return 0;
}