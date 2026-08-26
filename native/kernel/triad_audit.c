#include "triad_audit.h"
#include "triad_kernel.h"
#include "triad_serial.h"
#include "triad_mm.h"
#include "triad_disk.h"
#include "triad_vfs.h"
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <string.h>

static AuditLog audit_log = {0};
static int audit_initialized = 0;

static const char *event_names[AUDIT_MAX_EVENT] = {
    "PROC_CREATE",
    "PROC_EXIT",
    "PROC_KILL",
    "FILE_OPEN",
    "FILE_READ",
    "FILE_WRITE",
    "FILE_DELETE",
    "NET_CONNECT",
    "NET_SEND",
    "NET_RECV",
    "CAP_GRANT",
    "CAP_REVOKE",
    "CAP_CHECK_FAIL",
    "SYSCALL_DENIED",
    "STACK_CANARY_FAIL",
    "PAGE_FAULT",
    "GPF",
    "DOUBLE_FAULT",
    "EXEC_SIG_FAIL",
    "BINARY_LOAD",
    "MEMORY_OOM",
    "USER_SWITCH",
    "IPC_SEND",
    "IPC_RECV",
    "SCHED_YIELD",
    "SLEEP",
    "WAKE",
    "AI_QUERY",
    "AI_TOOL_USE",
    "SOLVER_RUN",
    "ML_LOAD",
    "ML_FORWARD",
    "DISK_MOUNT",
    "DISK_UNMOUNT",
    "DISK_ERROR",
    "KEY_PRESS"
};

static const char *severity_names[] = {
    "INFO",
    "WARN",
    "ERROR",
    "CRITICAL"
};

static uint64_t read_tsc(void) {
    uint64_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return (hi << 32) | lo;
}

void triad_audit_init(void) {
    if (audit_initialized) return;

    audit_log.head = 0;
    audit_log.tail = 0;
    audit_log.count = 0;
    audit_log.total_events = 0;
    audit_log.dropped_events = 0;
    audit_log.enabled = 1;
    audit_log.disk_log_enabled = 0;
    audit_log.disk_log_offset = 0;
    audit_log.disk_log_inode = 0;

    for (int i = 0; i < AUDIT_BUF_SIZE; i++) {
        audit_log.buffer[i].timestamp = 0;
        audit_log.buffer[i].pid = 0;
        audit_log.buffer[i].event_type = 0;
        audit_log.buffer[i].severity = 0;
        audit_log.buffer[i].details[0] = 0;
        for (int j = 0; j < AUDIT_MAX_EXTRA; j++) {
            audit_log.buffer[i].extra[j] = 0;
        }
    }

    audit_initialized = 1;
}

static void format_string(char *buf, size_t size, const char *fmt, va_list args) {
    if (!buf || size == 0) return;

    size_t written = 0;
    const char *p = fmt;

    while (*p && written < size - 1) {
        if (*p == '%' && *(p + 1)) {
            p++;
            int long_mod = 0;
            while (*p == 'l') {
                long_mod++;
                p++;
            }

            switch (*p) {
                case 'd':
                case 'i': {
                    int64_t val;
                    if (long_mod >= 1) {
                        val = va_arg(args, int64_t);
                    } else {
                        val = va_arg(args, int32_t);
                    }
                    if (val < 0) {
                        if (written < size - 1) buf[written++] = '-';
                        val = -val;
                    }
                    char tmp[24];
                    int tmp_idx = 0;
                    do {
                        tmp[tmp_idx++] = '0' + (val % 10);
                        val /= 10;
                    } while (val > 0 && tmp_idx < 24);
                    while (tmp_idx > 0 && written < size - 1) {
                        buf[written++] = tmp[--tmp_idx];
                    }
                    break;
                }
                case 'u':
                case 'x':
                case 'p': {
                    uint64_t val;
                    if (long_mod >= 1) {
                        val = va_arg(args, uint64_t);
                    } else {
                        val = va_arg(args, uint32_t);
                    }
                    if (*p == 'p') {
                        if (written < size - 1) buf[written++] = '0';
                        if (written < size - 1) buf[written++] = 'x';
                    }
                    char tmp[20];
                    int tmp_idx = 0;
                    int base = (*p == 'x' || *p == 'p') ? 16 : 10;
                    const char *digits = "0123456789abcdef";
                    do {
                        tmp[tmp_idx++] = digits[val % base];
                        val /= base;
                    } while (val > 0 && tmp_idx < 20);
                    while (tmp_idx > 0 && written < size - 1) {
                        buf[written++] = tmp[--tmp_idx];
                    }
                    break;
                }
                case 's': {
                    const char *s = va_arg(args, const char *);
                    if (!s) s = "(null)";
                    while (*s && written < size - 1) {
                        buf[written++] = *s++;
                    }
                    break;
                }
                case 'c': {
                    char c = (char)va_arg(args, int);
                    if (written < size - 1) buf[written++] = c;
                    break;
                }
                case '%':
                    if (written < size - 1) buf[written++] = '%';
                    break;
                default:
                    if (written < size - 1) buf[written++] = '%';
                    if (written < size - 1) buf[written++] = *p;
                    break;
            }
            p++;
        } else {
            if (written < size - 1) buf[written++] = *p++;
        }
    }

    buf[written] = 0;
}

void triad_audit_log(AuditEventType type, uint32_t severity, const char *fmt, ...) {
    if (!audit_initialized) triad_audit_init();
    if (!audit_log.enabled) return;

    if (audit_log.count >= AUDIT_BUF_SIZE) {
        audit_log.dropped_events++;
        return;
    }

    AuditEvent *ev = &audit_log.buffer[audit_log.head];

    ev->timestamp = read_tsc();
    ev->pid = 0;
    ev->event_type = (uint32_t)type;
    ev->severity = severity;

    va_list args;
    va_start(args, fmt);
    format_string(ev->details, AUDIT_MAX_DETAILS, fmt, args);
    va_end(args);

    for (int i = 0; i < AUDIT_MAX_EXTRA; i++) {
        ev->extra[i] = 0;
    }

    audit_log.head = (audit_log.head + 1) % AUDIT_BUF_SIZE;
    audit_log.count++;
    audit_log.total_events++;

    if (severity >= AUDIT_SEV_ERROR) {
        triad_serial_puts("[AUDIT] ");
        triad_serial_puts(triad_audit_severity_name((AuditSeverity)severity));
        triad_serial_puts(": ");
        triad_serial_puts(triad_audit_event_name(type));
        triad_serial_puts(" - ");
        triad_serial_puts(ev->details);
        triad_serial_puts("\n");
    }
}

int triad_audit_get_events(AuditEvent *out, int max) {
    if (!audit_initialized || !out || max <= 0) return 0;

    int retrieved = 0;
    while (audit_log.count > 0 && retrieved < max) {
        *out = audit_log.buffer[audit_log.tail];
        audit_log.tail = (audit_log.tail + 1) % AUDIT_BUF_SIZE;
        audit_log.count--;
        out++;
        retrieved++;
    }

    return retrieved;
}

void triad_audit_flush(void) {
    if (!audit_initialized) return;

    if (!audit_log.disk_log_enabled) return;

    if (audit_log.disk_log_inode == 0) {
        uint64_t ino = triad_disk_create_file("/var/log/audit.log");
        if (ino == 0) {
            triad_serial_puts("[AUDIT] cannot create audit log file on disk\n");
            return;
        }
        audit_log.disk_log_inode = ino;
    }

    AuditEvent events[64];
    int n = triad_audit_get_events(events, 64);
    if (n == 0) return;

    char line[512];
    for (int i = 0; i < n; i++) {
        triad_audit_format_event(&events[i], line, sizeof(line));
        size_t len = 0;
        while (line[len]) len++;
        line[len] = '\n';
        len++;
        triad_disk_write_file(audit_log.disk_log_inode, line, audit_log.disk_log_offset, len);
        audit_log.disk_log_offset += len;
    }

    triad_disk_sync();
}

void triad_audit_set_enabled(int enabled) {
    audit_log.enabled = enabled ? 1 : 0;
}

int triad_audit_is_enabled(void) {
    return audit_log.enabled;
}

void triad_audit_set_disk_log(int enabled) {
    audit_log.disk_log_enabled = enabled ? 1 : 0;
}

int triad_audit_get_disk_log(void) {
    return audit_log.disk_log_enabled;
}

uint64_t triad_audit_total_events(void) {
    return audit_log.total_events;
}

uint64_t triad_audit_dropped_events(void) {
    return audit_log.dropped_events;
}

void triad_audit_format_event(AuditEvent *ev, char *buf, size_t buf_size) {
    if (!ev || !buf || buf_size == 0) return;

    size_t written = 0;

    buf[written++] = '[';

    const char *sev = triad_audit_severity_name((AuditSeverity)ev->severity);
    while (*sev && written < buf_size - 1) {
        buf[written++] = *sev++;
    }
    buf[written++] = ']';
    buf[written++] = ' ';

    const char *name = triad_audit_event_name((AuditEventType)ev->event_type);
    while (*name && written < buf_size - 1) {
        buf[written++] = *name++;
    }

    if (written < buf_size - 4) {
        buf[written++] = ' ';
        buf[written++] = 'p';
        buf[written++] = 'i';
        buf[written++] = 'd';
        buf[written++] = '=';
        uint64_t pid = ev->pid;
        char tmp[24];
        int tmp_idx = 0;
        do {
            tmp[tmp_idx++] = '0' + (pid % 10);
            pid /= 10;
        } while (pid > 0 && tmp_idx < 24);
        while (tmp_idx > 0 && written < buf_size - 1) {
            buf[written++] = tmp[--tmp_idx];
        }
    }

    if (written < buf_size - 2) {
        buf[written++] = ':';
        buf[written++] = ' ';
    }

    const char *details = ev->details;
    while (*details && written < buf_size - 1) {
        buf[written++] = *details++;
    }

    buf[written] = 0;
}

const char *triad_audit_event_name(AuditEventType type) {
    if (type < AUDIT_MAX_EVENT) {
        return event_names[type];
    }
    return "UNKNOWN";
}

const char *triad_audit_severity_name(AuditSeverity sev) {
    if (sev <= AUDIT_SEV_CRITICAL) {
        return severity_names[sev];
    }
    return "UNKNOWN";
}
