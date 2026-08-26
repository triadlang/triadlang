#ifndef TRIAD_SANDBOX_H
#define TRIAD_SANDBOX_H

#include "triad_kernel.h"
#include "triad_vfs.h"

struct TriadSandbox {
    char     root[VFS_MAX_PATH];
    bool     allow_temp;
    bool     allow_home_read;
    char     home_read_dirs[MAX_CAPS][64];
    int      n_home_read;
    uint64_t cpu_time_limit_ms;
    uint64_t cpu_time_used_ms;
    uint64_t memory_limit_bytes;
    uint64_t memory_used_bytes;
    uint64_t network_rx_limit_bytes;
    uint64_t network_tx_limit_bytes;
    uint64_t network_rx_used;
    uint64_t network_tx_used;
    uint32_t max_fds;
    uint32_t max_threads;
    uint32_t max_processes;
};

bool triad_sb_check_path(TriadSandbox *sb, const char *path, bool write);
int  triad_sb_resolve(TriadSandbox *sb, const char *path, char *out, uint32_t out_len);
TriadSandbox *triad_sb_new(const char *root);

int triad_sb_check_cpu(TriadSandbox *sb, uint64_t elapsed_ms);
int triad_sb_check_memory(TriadSandbox *sb, uint64_t requested_bytes);
int triad_sb_check_network_rx(TriadSandbox *sb, uint64_t bytes);
int triad_sb_check_network_tx(TriadSandbox *sb, uint64_t bytes);

void triad_sb_set_cpu_limit(TriadSandbox *sb, uint64_t limit_ms);
void triad_sb_set_memory_limit(TriadSandbox *sb, uint64_t limit_bytes);
void triad_sb_set_network_rx_limit(TriadSandbox *sb, uint64_t limit_bytes);
void triad_sb_set_network_tx_limit(TriadSandbox *sb, uint64_t limit_bytes);
void triad_sb_set_max_fds(TriadSandbox *sb, uint32_t max_fds);
void triad_sb_set_max_threads(TriadSandbox *sb, uint32_t max_threads);

#endif
