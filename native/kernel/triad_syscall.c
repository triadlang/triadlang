#include "triad_kernel.h"
#include "triad_syscall.h"
#include "triad_vfs.h"
#include "triad_mm.h"
#include "triad_sched.h"
#include "triad_ai.h"
#include "triad_equilibrium.h"
#include "triad_sandbox.h"
#include "triad_disk.h"
#include "triad_net.h"
#include "triad_serial.h"
#include "triad_hardening.h"
#include "triad_audit.h"
#include "triad_aslr.h"
#include <string.h>

extern TriadProc triad_procs[MAX_PROCS];
extern int triad_current_pid;

static bool is_user_address(uint64_t addr) {
    return addr >= USER_SPACE_BASE && addr <= USER_SPACE_END;
}

static void *safe_user_ptr(uint64_t addr, size_t len) {
    if (!is_user_address(addr)) return NULL;
    if (addr + len < addr) return NULL;
    if (addr + len > USER_SPACE_END) return NULL;
    return (void *)addr;
}

int64_t triad_syscall_dispatch(TriadSyscall num,
                               uint64_t a, uint64_t b, uint64_t c,
                               uint64_t d, uint64_t e) {
    if (triad_current_pid < 0 || triad_current_pid >= MAX_PROCS) return -1;
    TriadProc *proc = &triad_procs[triad_current_pid];
    switch (num) {
    case SYS_EXIT: {
        TriadThread *t = triad_sched_current();
        if (t) triad_sched_exit(t->id);
        proc->alive = false;
        return 0;
    }
    case SYS_WRITE: {
        int fd = (int)a;
        const void *buf = safe_user_ptr(b, (uint32_t)c);
        uint32_t n = (uint32_t)c;
        if (!buf) return -1;
        if (fd == 1 || fd == 2) {
            triad_stac();
            triad_serial_puts((const char *)buf);
            triad_clac();
            return (int64_t)n;
        }
        triad_stac();
        int64_t result = triad_vfs_write(fd, buf, n);
        triad_clac();
        return result;
    }
    case SYS_READ: {
        int fd = (int)a;
        void *buf = safe_user_ptr(b, (uint32_t)c);
        uint32_t n = (uint32_t)c;
        if (!buf) return -1;
        triad_stac();
        int64_t result = triad_vfs_read(fd, buf, n);
        triad_clac();
        return result;
    }
    case SYS_OPEN: {
        const char *path = safe_user_ptr(a, 256);
        uint8_t mode = (uint8_t)b;
        if (!path) return -1;
        char resolved[VFS_MAX_PATH];
        triad_stac();
        char path_copy[VFS_MAX_PATH];
        int k = 0;
        while (k < VFS_MAX_PATH - 1) {
            char ch = ((volatile char *)path)[k];
            if (ch == 0) break;
            path_copy[k] = ch;
            k++;
        }
        path_copy[k] = 0;
        triad_clac();
        if (triad_sb_resolve(proc->sandbox, path_copy, resolved, VFS_MAX_PATH) < 0) {
            return -1;
        }
        if (!triad_sb_check_path(proc->sandbox, resolved, mode & 1)) {
            return -1;
        }
        return triad_vfs_open(resolved, mode);
    }
    case SYS_CLOSE: {
        return triad_vfs_close((int)a);
    }
    case SYS_MKDIR: {
        if (!(proc->caps.bits & CAP_FS_WRITE)) return -1;
        const char *path = safe_user_ptr(a, 256);
        if (!path) return -1;
        char resolved[VFS_MAX_PATH];
        triad_stac();
        char path_copy[VFS_MAX_PATH];
        int k = 0;
        while (k < VFS_MAX_PATH - 1) {
            char ch = ((volatile char *)path)[k];
            if (ch == 0) break;
            path_copy[k] = ch;
            k++;
        }
        path_copy[k] = 0;
        triad_clac();
        if (triad_sb_resolve(proc->sandbox, path_copy, resolved, VFS_MAX_PATH) < 0) {
            return -1;
        }
        return triad_vfs_mkdir(resolved);
    }
    case SYS_LISTDIR: {
        const char *path = safe_user_ptr(a, 256);
        char (*out)[VFS_MAX_PATH] = (char (*)[VFS_MAX_PATH])safe_user_ptr(b, (uint32_t)c * VFS_MAX_PATH);
        int max = (int)c;
        if (!path || !out) return -1;
        char resolved[VFS_MAX_PATH];
        triad_stac();
        char path_copy[VFS_MAX_PATH];
        int k = 0;
        while (k < VFS_MAX_PATH - 1) {
            char ch = ((volatile char *)path)[k];
            if (ch == 0) break;
            path_copy[k] = ch;
            k++;
        }
        path_copy[k] = 0;
        triad_clac();
        if (triad_sb_resolve(proc->sandbox, path_copy, resolved, VFS_MAX_PATH) < 0) {
            return -1;
        }
        char files[32][VFS_MAX_PATH];
        int n = triad_vfs_listdir(resolved, files, 32);
        if (n > max) n = max;
        triad_stac();
        for (int i = 0; i < n; i++) {
            int j = 0;
            while (files[i][j] && j < VFS_MAX_PATH) {
                out[i][j] = files[i][j];
                j++;
            }
            out[i][j] = 0;
        }
        triad_clac();
        return n;
    }
    case SYS_REMOVE: {
        if (!(proc->caps.bits & CAP_FS_REMOVE)) return -1;
        const char *path = safe_user_ptr(a, 256);
        if (!path) return -1;
        char resolved[VFS_MAX_PATH];
        triad_stac();
        char path_copy[VFS_MAX_PATH];
        int k = 0;
        while (k < VFS_MAX_PATH - 1) {
            char ch = ((volatile char *)path)[k];
            if (ch == 0) break;
            path_copy[k] = ch;
            k++;
        }
        path_copy[k] = 0;
        triad_clac();
        if (triad_sb_resolve(proc->sandbox, path_copy, resolved, VFS_MAX_PATH) < 0) {
            return -1;
        }
        return triad_vfs_remove(resolved);
    }
    case SYS_GETPID: {
        return triad_current_pid;
    }
    case SYS_GETTICKS: {
        return (int64_t)triad_sched_get_ticks();
    }
    case SYS_SLEEP: {
        TriadThread *t = triad_sched_current();
        if (t) triad_sched_block(t->id);
        return 0;
    }
    case SYS_YIELD: {
        triad_sched_yield();
        return 0;
    }
    case SYS_CAP_CHECK: {
        uint32_t cap_bit = (uint32_t)a;
        return (proc->caps.bits & cap_bit) ? 1 : 0;
    }
    case SYS_AI_QUERY: {
        const char *query = safe_user_ptr(a, 512);
        char *out = safe_user_ptr(b, (uint32_t)c);
        uint32_t out_len = (uint32_t)c;
        if (!query || !out) return -1;
        char query_copy[512];
        triad_stac();
        int k = 0;
        while (k < 511) {
            char ch = ((volatile char *)query)[k];
            if (ch == 0) break;
            query_copy[k] = ch;
            k++;
        }
        query_copy[k] = 0;
        triad_clac();
        char result_buf[1024];
        int n = triad_ai_query(query_copy, result_buf, sizeof(result_buf));
        triad_stac();
        int j = 0;
        while (j < (int)out_len - 1 && result_buf[j]) {
            out[j] = result_buf[j];
            j++;
        }
        out[j] = 0;
        triad_clac();
        return n;
    }
    case SYS_AI_RECORD: {
        const char *source = safe_user_ptr(a, 128);
        const char *text = safe_user_ptr(b, 512);
        if (!source || !text) return -1;
        char src_copy[128];
        char txt_copy[512];
        triad_stac();
        int k = 0;
        while (k < 127) {
            char ch = ((volatile char *)source)[k];
            if (ch == 0) break;
            src_copy[k] = ch;
            k++;
        }
        src_copy[k] = 0;
        k = 0;
        while (k < 511) {
            char ch = ((volatile char *)text)[k];
            if (ch == 0) break;
            txt_copy[k] = ch;
            k++;
        }
        txt_copy[k] = 0;
        triad_clac();
        triad_ai_observe(triad_current_pid, src_copy, txt_copy);
        return 0;
    }
    case SYS_AI_TOOL_USE: {
        const char *tool = safe_user_ptr(a, 128);
        const char *args = safe_user_ptr(b, 512);
        char *out = safe_user_ptr(c, (uint32_t)d);
        uint32_t out_len = (uint32_t)d;
        bool confirm = (bool)e;
        if (!tool || !args || !out) return -1;
        if (!(proc->caps.bits & CAP_AI_TOOL)) return -1;
        char tool_copy[128];
        char args_copy[512];
        triad_stac();
        int k = 0;
        while (k < 127) {
            char ch = ((volatile char *)tool)[k];
            if (ch == 0) break;
            tool_copy[k] = ch;
            k++;
        }
        tool_copy[k] = 0;
        k = 0;
        while (k < 511) {
            char ch = ((volatile char *)args)[k];
            if (ch == 0) break;
            args_copy[k] = ch;
            k++;
        }
        args_copy[k] = 0;
        triad_clac();
        char result_buf[1024];
        int n = triad_ai_tool_use(triad_current_pid, tool_copy, args_copy, result_buf, sizeof(result_buf), confirm);
        triad_stac();
        int j = 0;
        while (j < (int)out_len - 1 && result_buf[j]) {
            out[j] = result_buf[j];
            j++;
        }
        out[j] = 0;
        triad_clac();
        return n;
    }
    case SYS_SEEK: {
        int fd = (int)a;
        uint32_t offset = (uint32_t)b;
        return triad_vfs_seek(fd, offset);
    }
    case SYS_STAT: {
        const char *path = safe_user_ptr(a, 256);
        TriadVfsNode *out = (TriadVfsNode *)safe_user_ptr(b, sizeof(TriadVfsNode));
        if (!path || !out) return -1;
        char resolved[VFS_MAX_PATH];
        triad_stac();
        char path_copy[VFS_MAX_PATH];
        int k = 0;
        while (k < VFS_MAX_PATH - 1) {
            char ch = ((volatile char *)path)[k];
            if (ch == 0) break;
            path_copy[k] = ch;
            k++;
        }
        path_copy[k] = 0;
        triad_clac();
        if (triad_sb_resolve(proc->sandbox, path_copy, resolved, VFS_MAX_PATH) < 0) {
            return -1;
        }
        TriadVfsNode node;
        int rc = triad_vfs_stat(resolved, &node);
        if (rc == 0) {
            triad_stac();
            *out = node;
            triad_clac();
        }
        return rc;
    }
    case SYS_CAP_GRANT: {
        int target_pid = (int)a;
        uint32_t cap_bit = (uint32_t)b;
        if (target_pid < 0 || target_pid >= MAX_PROCS) return -1;
        if (!(proc->caps.bits & CAP_KERNEL)) return -1;
        triad_procs[target_pid].caps.bits |= cap_bit;
        return 0;
    }
    case SYS_DISK_SYNC: {
        return triad_disk_sync_vfs();
    }
    case SYS_DISK_MOUNT: {
        uint64_t start_sector = (uint64_t)a;
        return triad_disk_mount(start_sector);
    }
    case SYS_NET_SOCKET: {
        if (!(proc->caps.bits & (CAP_NET_GET | CAP_NET_POST))) return -1;
        int if_id = (int)a;
        uint16_t port = (uint16_t)b;
        int sock_id;
        return triad_net_udp_socket(if_id, port, &sock_id);
    }
    case SYS_NET_CONNECT: {
        if (!(proc->caps.bits & CAP_NET_POST)) return -1;
        int if_id = (int)a;
        const char *dest_str = safe_user_ptr(b, 16);
        if (!dest_str) return -1;
        NetIp4 dest;
        triad_stac();
        char dest_copy[16];
        int k = 0;
        while (k < 15) {
            char ch = ((volatile char *)dest_str)[k];
            if (ch == 0) break;
            dest_copy[k] = ch;
            k++;
        }
        dest_copy[k] = 0;
        triad_clac();
        net_str_to_ip4(dest_copy, &dest);
        uint16_t port = (uint16_t)c;
        int sock_id;
        return triad_net_tcp_connect(if_id, dest, port, &sock_id);
    }
    case SYS_NET_LISTEN: {
        if (!(proc->caps.bits & CAP_NET_GET)) return -1;
        int if_id = (int)a;
        uint16_t port = (uint16_t)b;
        int sock_id;
        return triad_net_tcp_listen(if_id, port, &sock_id);
    }
    case SYS_NET_SEND: {
        if (!(proc->caps.bits & CAP_NET_POST)) return -1;
        int sock_id = (int)a;
        const uint8_t *data = (const uint8_t *)safe_user_ptr(b, (uint16_t)c);
        uint16_t len = (uint16_t)c;
        if (!data) return -1;
        uint8_t data_copy[4096];
        if (len > sizeof(data_copy)) return -1;
        triad_stac();
        for (uint16_t i = 0; i < len; i++) {
            data_copy[i] = ((volatile uint8_t *)data)[i];
        }
        triad_clac();
        return triad_net_tcp_send(sock_id, data_copy, len);
    }
    case SYS_NET_RECV: {
        if (!(proc->caps.bits & CAP_NET_GET)) return -1;
        int sock_id = (int)a;
        uint8_t *buf = (uint8_t *)safe_user_ptr(b, (uint16_t)c);
        uint16_t max = (uint16_t)c;
        if (!buf) return -1;
        uint8_t recv_buf[4096];
        int n = triad_net_tcp_recv(sock_id, recv_buf, max > sizeof(recv_buf) ? sizeof(recv_buf) : max);
        if (n > 0) {
            triad_stac();
            for (int i = 0; i < n; i++) {
                buf[i] = recv_buf[i];
            }
            triad_clac();
        }
        return n;
    }
    case SYS_EQUILIBRIUM: {
        EqHwDomain domain = (EqHwDomain)a;
        double u = 0.0;
        triad_eq_read(domain, &u);
        return (int64_t)(u * 10000.0);
    }
    case SYS_SOLVER: {
        return 0;
    }
    case SYS_NET_GET: {
        if (!(proc->caps.bits & CAP_NET_GET)) return -1;
        int sock_id = (int)a;
        uint8_t *buf = (uint8_t *)safe_user_ptr(b, (uint16_t)c);
        uint16_t max = (uint16_t)c;
        if (!buf) return -1;
        NetIp4 src;
        uint16_t src_port;
        uint8_t recv_buf[4096];
        int n = triad_net_udp_recv(sock_id, recv_buf, max > sizeof(recv_buf) ? sizeof(recv_buf) : max, &src, &src_port);
        if (n > 0) {
            triad_stac();
            for (int i = 0; i < n; i++) {
                buf[i] = recv_buf[i];
            }
            triad_clac();
        }
        return n;
    }
    case SYS_NET_POST: {
        if (!(proc->caps.bits & CAP_NET_POST)) return -1;
        int sock_id = (int)a;
        const char *dest_str = safe_user_ptr(b, 16);
        if (!dest_str) return -1;
        NetIp4 dest;
        triad_stac();
        char dest_copy[16];
        int k = 0;
        while (k < 15) {
            char ch = ((volatile char *)dest_str)[k];
            if (ch == 0) break;
            dest_copy[k] = ch;
            k++;
        }
        dest_copy[k] = 0;
        triad_clac();
        net_str_to_ip4(dest_copy, &dest);
        uint16_t port = (uint16_t)c;
        const uint8_t *data = (const uint8_t *)safe_user_ptr(d, (uint16_t)e);
        uint16_t len = (uint16_t)e;
        if (!data) return -1;
        uint8_t data_copy[4096];
        if (len > sizeof(data_copy)) return -1;
        triad_stac();
        for (uint16_t i = 0; i < len; i++) {
            data_copy[i] = ((volatile uint8_t *)data)[i];
        }
        triad_clac();
        return triad_net_udp_send(sock_id, dest, port, data_copy, len);
    }
    case SYS_ML_LOAD: {
        if (!(proc->caps.bits & CAP_ML_LOCAL)) return -1;
        return -1;
    }
    case SYS_ML_FORWARD: {
        if (!(proc->caps.bits & CAP_ML_LOCAL)) return -1;
        return -1;
    }
    case SYS_BRK: {
        return 0;
    }
    case SYS_GETENV: {
        return 0;
    }
    case SYS_SPAWN: {
        return -1;
    }
    case SYS_WAIT: {
        return -1;
    }
    case SYS_PIPE: {
        return -1;
    }
    case SYS_DUP: {
        return -1;
    }
    case SYS_MMAP: {
        return -1;
    }
    case SYS_SETENV: {
        return -1;
    }
    default:
        triad_audit_log(AUDIT_SYSCALL_DENIED, AUDIT_SEV_WARN,
                        "unknown syscall %d from pid=%d", (int)num, triad_current_pid);
        return -1;
    }
}

int triad_syscall_init(void) {
    return 0;
}

void triad_syscall_test(void) {
    triad_ai_observe(0, "kernel", "syscall test observation");
    char buf[512];
    int n = triad_ai_query("test", buf, sizeof(buf));
    (void)n;
}
