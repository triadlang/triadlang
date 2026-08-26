#include "triad_kernel.h"
#include "triad_sandbox.h"
#include "triad_mm.h"

bool triad_sb_check_path(TriadSandbox *sb, const char *path, bool write) {
    (void)write;
    if (!sb || !path) return false;
    int k = 0;
    int j = 0;
    while (sb->root[j] && path[k] == sb->root[j]) {
        k++;
        j++;
    }
    if (sb->root[j] == 0) return true;
    if (path[0] == '/') {
        if (sb->allow_temp) {
            const char tmp[] = "/tmp";
            int i = 0;
            while (tmp[i] && path[i] == tmp[i]) i++;
            if (tmp[i] == 0) return true;
        }
        if (sb->allow_home_read) {
            const char home[] = "/home";
            int i = 0;
            while (home[i] && path[i] == home[i]) i++;
            if (home[i] == 0) return true;
        }
    }
    return false;
}

int triad_sb_resolve(TriadSandbox *sb, const char *path, char *out, uint32_t out_len) {
    if (!sb || !path || !out) return -1;
    if (path[0] == '/') {
        int k = 0;
        while (path[k] && k < (int)out_len - 1) {
            if (path[k] == '.' && path[k + 1] == '.') return -1;
            out[k] = path[k];
            k++;
        }
        out[k] = 0;
    } else {
        int k = 0;
        int j = 0;
        while (sb->root[j] && k < (int)out_len - 1) {
            out[k] = sb->root[j];
            k++; j++;
        }
        if (k > 0 && out[k - 1] != '/' && k < (int)out_len - 1) {
            out[k] = '/';
            k++;
        }
        int i = 0;
        while (path[i] && k < (int)out_len - 1) {
            if (path[i] == '.' && path[i + 1] == '.') return -1;
            out[k] = path[i];
            k++; i++;
        }
        out[k] = 0;
    }
    return 0;
}

TriadSandbox *triad_sb_new(const char *root) {
    TriadSandbox *sb = (TriadSandbox *)triad_mm_alloc(sizeof(TriadSandbox));
    if (!sb) return NULL;
    int k = 0;
    while (root[k] && k < VFS_MAX_PATH - 1) {
        sb->root[k] = root[k];
        k++;
    }
    sb->root[k] = 0;
    sb->allow_temp = true;
    sb->allow_home_read = false;
    sb->n_home_read = 0;
    sb->cpu_time_limit_ms = 0;
    sb->cpu_time_used_ms = 0;
    sb->memory_limit_bytes = 0;
    sb->memory_used_bytes = 0;
    sb->network_rx_limit_bytes = 0;
    sb->network_tx_limit_bytes = 0;
    sb->network_rx_used = 0;
    sb->network_tx_used = 0;
    sb->max_fds = 32;
    sb->max_threads = 8;
    sb->max_processes = 4;
    return sb;
}

int triad_sb_check_cpu(TriadSandbox *sb, uint64_t elapsed_ms) {
    if (!sb) return 1;
    if (sb->cpu_time_limit_ms == 0) return 1;
    sb->cpu_time_used_ms += elapsed_ms;
    if (sb->cpu_time_used_ms > sb->cpu_time_limit_ms) return 0;
    return 1;
}

int triad_sb_check_memory(TriadSandbox *sb, uint64_t requested_bytes) {
    if (!sb) return 1;
    if (sb->memory_limit_bytes == 0) return 1;
    if (sb->memory_used_bytes + requested_bytes > sb->memory_limit_bytes) return 0;
    sb->memory_used_bytes += requested_bytes;
    return 1;
}

int triad_sb_check_network_rx(TriadSandbox *sb, uint64_t bytes) {
    if (!sb) return 1;
    if (sb->network_rx_limit_bytes == 0) return 1;
    if (sb->network_rx_used + bytes > sb->network_rx_limit_bytes) return 0;
    sb->network_rx_used += bytes;
    return 1;
}

int triad_sb_check_network_tx(TriadSandbox *sb, uint64_t bytes) {
    if (!sb) return 1;
    if (sb->network_tx_limit_bytes == 0) return 1;
    if (sb->network_tx_used + bytes > sb->network_tx_limit_bytes) return 0;
    sb->network_tx_used += bytes;
    return 1;
}

void triad_sb_set_cpu_limit(TriadSandbox *sb, uint64_t limit_ms) {
    if (sb) sb->cpu_time_limit_ms = limit_ms;
}

void triad_sb_set_memory_limit(TriadSandbox *sb, uint64_t limit_bytes) {
    if (sb) sb->memory_limit_bytes = limit_bytes;
}

void triad_sb_set_network_rx_limit(TriadSandbox *sb, uint64_t limit_bytes) {
    if (sb) sb->network_rx_limit_bytes = limit_bytes;
}

void triad_sb_set_network_tx_limit(TriadSandbox *sb, uint64_t limit_bytes) {
    if (sb) sb->network_tx_limit_bytes = limit_bytes;
}

void triad_sb_set_max_fds(TriadSandbox *sb, uint32_t max_fds) {
    if (sb) sb->max_fds = max_fds;
}

void triad_sb_set_max_threads(TriadSandbox *sb, uint32_t max_threads) {
    if (sb) sb->max_threads = max_threads;
}
