#include "triad_disk.h"
#include "triad_ata.h"
#include "triad_mm.h"
#include "triad_vfs.h"
#include "triad_serial.h"
#include <string.h>

static DiskFsSuperblock disk_super;
static uint8_t *block_bitmap = NULL;
static uint64_t bitmap_blocks = 0;
static uint64_t mounted_start = 0;
static bool disk_mounted = false;

static uint8_t *block_cache[DISK_CACHE_SIZE];
static uint64_t block_cache_tags[DISK_CACHE_SIZE];
static int block_cache_valid[DISK_CACHE_SIZE];

static int disk_read_sector(uint64_t lba, void *buf) {
    return triad_ata_read(0, 0, mounted_start + lba, buf, 1);
}

static int disk_write_sector(uint64_t lba, const void *buf) {
    return triad_ata_write(0, 0, mounted_start + lba, buf, 1);
}

static void cache_init(void) {
    for (int i = 0; i < DISK_CACHE_SIZE; i++) {
        block_cache[i] = (uint8_t *)triad_mm_alloc(DISK_SECTOR_SIZE);
        block_cache_tags[i] = 0xFFFFFFFFFFFFFFFFULL;
        block_cache_valid[i] = 0;
    }
}

static int cache_read(uint64_t block, void *buf) {
    for (int i = 0; i < DISK_CACHE_SIZE; i++) {
        if (block_cache_valid[i] && block_cache_tags[i] == block) {
            memcpy(buf, block_cache[i], DISK_SECTOR_SIZE);
            return 0;
        }
    }
    if (disk_read_sector(block, buf) < 0) return -1;
    int idx = 0;
    for (int i = 1; i < DISK_CACHE_SIZE; i++) {
        if (!block_cache_valid[i]) {
            idx = i;
            break;
        }
    }
    if (!block_cache[idx]) return 0;
    memcpy(block_cache[idx], buf, DISK_SECTOR_SIZE);
    block_cache_tags[idx] = block;
    block_cache_valid[idx] = 1;
    return 0;
}

static int cache_write(uint64_t block, const void *buf) {
    if (!buf) return -1;
    for (int i = 0; i < DISK_CACHE_SIZE; i++) {
        if (block_cache_valid[i] && block_cache_tags[i] == block && block_cache[i]) {
            memcpy(block_cache[i], buf, DISK_SECTOR_SIZE);
        }
    }
    return disk_write_sector(block, buf);
}

int triad_disk_init(void) {
    cache_init();
    disk_mounted = false;
    return 0;
}

int triad_disk_format(uint64_t start_sector, uint64_t num_sectors) {
    /* mesma convenção de triad_disk_mount: escritas relativas à base. */
    mounted_start = start_sector;
    DiskFsSuperblock sb;
    memset(&sb, 0, sizeof(sb));
    sb.magic = DISKFS_MAGIC;
    sb.version = DISKFS_VERSION;
    sb.total_blocks = num_sectors;
    sb.free_blocks = num_sectors - 1;
    sb.inode_count = 1;
    sb.root_inode = 1;
    sb.bitmap_start = 1;
    sb.bitmap_size = (num_sectors + 7) / 8 / DISK_SECTOR_SIZE + 1;
    memcpy(sb.label, "TriadDisk", 10);

    uint8_t sector[DISK_SECTOR_SIZE];
    memset(sector, 0, DISK_SECTOR_SIZE);
    memcpy(sector, &sb, sizeof(sb));
    if (disk_write_sector(0, sector) < 0) return -1;

    uint64_t bitmap_bytes = (num_sectors + 7) / 8;
    block_bitmap = (uint8_t *)triad_mm_alloc(bitmap_bytes);
    memset(block_bitmap, 0, bitmap_bytes);
    block_bitmap[0] |= 1;

    for (uint64_t i = 0; i < sb.bitmap_size; i++) {
        memset(sector, 0, DISK_SECTOR_SIZE);
        memcpy(sector, block_bitmap + i * DISK_SECTOR_SIZE, 
               bitmap_bytes - i * DISK_SECTOR_SIZE > DISK_SECTOR_SIZE 
               ? DISK_SECTOR_SIZE 
               : bitmap_bytes - i * DISK_SECTOR_SIZE);
        if (disk_write_sector(sb.bitmap_start + i, sector) < 0) return -1;
    }
    bitmap_blocks = sb.bitmap_size;

    DiskFsInode root;
    memset(&root, 0, sizeof(root));
    root.ino = 1;
    root.parent_ino = 0;
    root.type = DISKFS_TYPE_DIR;
    root.mode = 0755;
    root.size = 0;
    root.created = 0;
    root.modified = 0;
    root.first_block = 0;
    memcpy(root.name, "/", 2);

    memset(sector, 0, DISK_SECTOR_SIZE);
    memcpy(sector, &root, sizeof(root));
    if (disk_write_sector(sb.bitmap_start + sb.bitmap_size, sector) < 0) return -1;

    disk_super = sb;
    disk_mounted = true;
    return 0;
}

int triad_disk_mount(uint64_t start_sector) {
    mounted_start = start_sector;

    uint8_t sector[DISK_SECTOR_SIZE];
    if (disk_read_sector(0, sector) < 0) return -1;

    memcpy(&disk_super, sector, sizeof(DiskFsSuperblock));
    if (disk_super.magic != DISKFS_MAGIC) return -1;
    if (disk_super.version != DISKFS_VERSION) return -1;

    uint64_t bitmap_bytes = (disk_super.total_blocks + 7) / 8;
    block_bitmap = (uint8_t *)triad_mm_alloc(bitmap_bytes);
    if (!block_bitmap) return -1;

    for (uint64_t i = 0; i < disk_super.bitmap_size; i++) {
        if (disk_read_sector(disk_super.bitmap_start + i, sector) < 0) return -1;
        memcpy(block_bitmap + i * DISK_SECTOR_SIZE, sector, DISK_SECTOR_SIZE);
    }

    bitmap_blocks = disk_super.bitmap_size;
    disk_mounted = true;
    return 0;
}

int triad_disk_read_block(uint64_t block, void *buf) {
    return cache_read(block, buf);
}

int triad_disk_write_block(uint64_t block, const void *buf) {
    return cache_write(block, buf);
}

uint64_t triad_disk_alloc_block(void) {
    if (!disk_mounted) return 0;
    
    uint64_t bitmap_bytes = (disk_super.total_blocks + 7) / 8;
    for (uint64_t i = 0; i < bitmap_bytes; i++) {
        if (block_bitmap[i] == 0xFF) continue;
        for (int b = 0; b < 8; b++) {
            if (!(block_bitmap[i] & (1 << b))) {
                uint64_t block = i * 8 + b;
                block_bitmap[i] |= (1 << b);
                disk_super.free_blocks--;
                return block;
            }
        }
    }
    return 0;
}

void triad_disk_free_block(uint64_t block) {
    if (!disk_mounted) return;
    uint64_t byte = block / 8;
    uint8_t bit = block % 8;
    block_bitmap[byte] &= ~(1 << bit);
    disk_super.free_blocks++;
}

static uint64_t inode_sector(uint64_t ino) {
    return disk_super.bitmap_start + disk_super.bitmap_size + (ino - 1);
}

uint64_t triad_disk_create_file(const char *path) {
    if (!disk_mounted) return 0;

    uint8_t sector[DISK_SECTOR_SIZE];
    uint64_t ino = ++disk_super.inode_count;
    uint64_t parent_ino = 1;

    DiskFsInode inode;
    memset(&inode, 0, sizeof(inode));
    inode.ino = ino;
    inode.parent_ino = parent_ino;
    inode.type = DISKFS_TYPE_FILE;
    inode.mode = 0644;
    inode.size = 0;
    inode.first_block = 0;

    const char *name = path;
    for (const char *p = path; *p; p++) {
        if (*p == '/') name = p + 1;
    }
    int k = 0;
    while (name[k] && k < 63) {
        inode.name[k] = name[k];
        k++;
    }
    inode.name[k] = 0;

    memset(sector, 0, DISK_SECTOR_SIZE);
    memcpy(sector, &inode, sizeof(inode));
    if (disk_write_sector(inode_sector(ino), sector) < 0) return 0;

    return ino;
}

uint64_t triad_disk_create_dir(const char *path) {
    if (!disk_mounted) return 0;

    uint8_t sector[DISK_SECTOR_SIZE];
    uint64_t ino = ++disk_super.inode_count;

    DiskFsInode inode;
    memset(&inode, 0, sizeof(inode));
    inode.ino = ino;
    inode.parent_ino = 1;
    inode.type = DISKFS_TYPE_DIR;
    inode.mode = 0755;
    inode.size = 0;
    inode.first_block = 0;

    const char *name = path;
    for (const char *p = path; *p; p++) {
        if (*p == '/') name = p + 1;
    }
    int k = 0;
    while (name[k] && k < 63) {
        inode.name[k] = name[k];
        k++;
    }
    inode.name[k] = 0;

    memset(sector, 0, DISK_SECTOR_SIZE);
    memcpy(sector, &inode, sizeof(inode));
    if (disk_write_sector(inode_sector(ino), sector) < 0) return 0;

    return ino;
}

uint64_t triad_disk_lookup(const char *path) {
    if (!disk_mounted) return 0;

    uint8_t sector[DISK_SECTOR_SIZE];
    for (uint64_t ino = 1; ino <= disk_super.inode_count; ino++) {
        if (disk_read_sector(inode_sector(ino), sector) < 0) continue;
        DiskFsInode inode;
        memcpy(&inode, sector, sizeof(inode));
        if (inode.type == 0) continue;
        
        char full_path[256];
        int pos = 0;
        if (inode.parent_ino > 0 && inode.parent_ino != ino) {
            full_path[pos++] = '/';
        }
        int k = 0;
        while (inode.name[k] && pos < 255) {
            full_path[pos++] = inode.name[k++];
        }
        full_path[pos] = 0;

        if (strcmp(path, full_path) == 0) return ino;
        if (strcmp(path, inode.name) == 0) return ino;
    }
    return 0;
}

int triad_disk_read_file(uint64_t ino, void *buf, uint64_t offset, uint64_t size) {
    if (!disk_mounted) return -1;

    uint8_t sector[DISK_SECTOR_SIZE];
    if (disk_read_sector(inode_sector(ino), sector) < 0) return -1;

    DiskFsInode inode;
    memcpy(&inode, sector, sizeof(inode));
    if (inode.type != DISKFS_TYPE_FILE) return -1;

    if (offset >= inode.size) return 0;
    uint64_t to_read = size;
    if (offset + size > inode.size) {
        to_read = inode.size - offset;
    }

    uint64_t block_idx = offset / DISK_SECTOR_SIZE;
    uint64_t block_offset = offset % DISK_SECTOR_SIZE;
    uint64_t read_pos = 0;

    while (read_pos < to_read) {
        uint64_t block = inode.first_block + block_idx;
        if (cache_read(block, sector) < 0) return -1;

        uint64_t chunk = DISK_SECTOR_SIZE - block_offset;
        if (chunk > to_read - read_pos) chunk = to_read - read_pos;

        memcpy((uint8_t *)buf + read_pos, sector + block_offset, chunk);
        read_pos += chunk;
        block_idx++;
        block_offset = 0;
    }

    return (int)to_read;
}

int triad_disk_write_file(uint64_t ino, const void *buf, uint64_t offset, uint64_t size) {
    if (!disk_mounted) return -1;

    uint8_t sector[DISK_SECTOR_SIZE];
    if (disk_read_sector(inode_sector(ino), sector) < 0) return -1;

    DiskFsInode inode;
    memcpy(&inode, sector, sizeof(inode));
    if (inode.type != DISKFS_TYPE_FILE) return -1;

    if (inode.first_block == 0) {
        inode.first_block = triad_disk_alloc_block();
        if (inode.first_block == 0) return -1;
    }

    uint64_t block_idx = offset / DISK_SECTOR_SIZE;
    uint64_t block_offset = offset % DISK_SECTOR_SIZE;
    uint64_t write_pos = 0;

    while (write_pos < size) {
        uint64_t block = inode.first_block + block_idx;
        
        if (block_offset > 0 || size - write_pos < DISK_SECTOR_SIZE) {
            if (cache_read(block, sector) < 0) return -1;
        }

        uint64_t chunk = DISK_SECTOR_SIZE - block_offset;
        if (chunk > size - write_pos) chunk = size - write_pos;

        memcpy(sector + block_offset, (uint8_t *)buf + write_pos, chunk);
        if (cache_write(block, sector) < 0) return -1;

        write_pos += chunk;
        block_idx++;
        block_offset = 0;
    }

    inode.size = offset + size;
    inode.modified = 0;
    memset(sector, 0, DISK_SECTOR_SIZE);
    memcpy(sector, &inode, sizeof(inode));
    if (disk_write_sector(inode_sector(ino), sector) < 0) return -1;

    return (int)size;
}

int triad_disk_list_dir(uint64_t ino, DiskFsDirEntry *entries, int max) {
    if (!disk_mounted) return -1;

    uint8_t sector[DISK_SECTOR_SIZE];
    int count = 0;

    for (uint64_t i = 1; i <= disk_super.inode_count && count < max; i++) {
        if (i == ino) continue;
        if (disk_read_sector(inode_sector(i), sector) < 0) continue;

        DiskFsInode inode;
        memcpy(&inode, sector, sizeof(inode));
        if (inode.type == 0) continue;
        if (inode.parent_ino != ino) continue;

        entries[count].ino = inode.ino;
        entries[count].type = (uint8_t)inode.type;
        int k = 0;
        while (inode.name[k] && k < 63) {
            entries[count].name[k] = inode.name[k];
            k++;
        }
        entries[count].name[k] = 0;
        count++;
    }

    return count;
}

int triad_disk_remove(uint64_t ino) {
    if (!disk_mounted) return -1;

    uint8_t sector[DISK_SECTOR_SIZE];
    memset(sector, 0, DISK_SECTOR_SIZE);
    if (disk_write_sector(inode_sector(ino), sector) < 0) return -1;

    return 0;
}

void triad_disk_sync(void) {
    if (!disk_mounted) return;

    uint8_t sector[DISK_SECTOR_SIZE];
    memset(sector, 0, DISK_SECTOR_SIZE);
    memcpy(sector, &disk_super, sizeof(disk_super));
    disk_write_sector(0, sector);

    for (uint64_t i = 0; i < disk_super.bitmap_size; i++) {
        memset(sector, 0, DISK_SECTOR_SIZE);
        memcpy(sector, block_bitmap + i * DISK_SECTOR_SIZE, DISK_SECTOR_SIZE);
        disk_write_sector(disk_super.bitmap_start + i, sector);
    }
}

int triad_disk_sync_vfs(void) {
    if (!disk_mounted) return -1;

    char files[32][256];
    int n = triad_vfs_listdir("/", files, 32);

    for (int i = 0; i < n; i++) {
        char path[512];
        int k = 0;
        path[k++] = '/';
        int j = 0;
        while (files[i][j] && k < 510) {
            path[k++] = files[i][j++];
        }
        path[k] = 0;

        TriadVfsNode node;
        if (triad_vfs_stat(path, &node) < 0) continue;

        uint64_t ino = triad_disk_create_file(path);
        if (ino == 0) continue;

        if (node.data && node.size > 0) {
            triad_disk_write_file(ino, node.data, 0, node.size);
        }
    }

    triad_disk_sync();
    return 0;
}

int triad_disk_load_vfs(void) {
    if (!disk_mounted) return -1;

    uint8_t sector[DISK_SECTOR_SIZE];
    char buf[4096];

    for (uint64_t ino = 2; ino <= disk_super.inode_count; ino++) {
        if (disk_read_sector(inode_sector(ino), sector) < 0) continue;

        DiskFsInode inode;
        memcpy(&inode, sector, sizeof(inode));
        if (inode.type == 0 || inode.type == DISKFS_TYPE_DIR) continue;

        char path[256];
        path[0] = '/';
        int k = 0;
        while (inode.name[k] && k < 254) {
            path[k + 1] = inode.name[k];
            k++;
        }
        path[k + 1] = 0;

        triad_vfs_create(path);

        uint64_t to_read = inode.size;
        uint64_t offset = 0;
        while (to_read > 0) {
            uint64_t chunk = to_read > 4096 ? 4096 : to_read;
            int n = triad_disk_read_file(ino, buf, offset, chunk);
            if (n <= 0) break;
            int fd = triad_vfs_open(path, 1);
            if (fd < 0) break;
            triad_vfs_write(fd, buf, n);
            triad_vfs_close(fd);
            offset += n;
            to_read -= n;
        }
    }

    return 0;
}