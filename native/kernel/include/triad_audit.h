#ifndef TRIAD_AUDIT_H
#define TRIAD_AUDIT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef enum {
    AUDIT_PROC_CREATE = 0,
    AUDIT_PROC_EXIT = 1,
    AUDIT_PROC_KILL = 2,
    AUDIT_FILE_OPEN = 3,
    AUDIT_FILE_READ = 4,
    AUDIT_FILE_WRITE = 5,
    AUDIT_FILE_DELETE = 6,
    AUDIT_NET_CONNECT = 7,
    AUDIT_NET_SEND = 8,
    AUDIT_NET_RECV = 9,
    AUDIT_CAP_GRANT = 10,
    AUDIT_CAP_REVOKE = 11,
    AUDIT_CAP_CHECK_FAIL = 12,
    AUDIT_SYSCALL_DENIED = 13,
    AUDIT_STACK_CANARY_FAIL = 14,
    AUDIT_PAGE_FAULT = 15,
    AUDIT_GPF = 16,
    AUDIT_DOUBLE_FAULT = 17,
    AUDIT_EXEC_SIG_FAIL = 18,
    AUDIT_BINARY_LOAD = 19,
    AUDIT_MEMORY_OOM = 20,
    AUDIT_USER_SWITCH = 21,
    AUDIT_IPC_SEND = 22,
    AUDIT_IPC_RECV = 23,
    AUDIT_SCHED_YIELD = 24,
    AUDIT_SLEEP = 25,
    AUDIT_WAKE = 26,
    AUDIT_AI_QUERY = 27,
    AUDIT_AI_TOOL_USE = 28,
    AUDIT_SOLVER_RUN = 29,
    AUDIT_ML_LOAD = 30,
    AUDIT_ML_FORWARD = 31,
    AUDIT_DISK_MOUNT = 32,
    AUDIT_DISK_UNMOUNT = 33,
    AUDIT_DISK_ERROR = 34,
    AUDIT_KEY_PRESS = 35,
    AUDIT_MAX_EVENT = 36
} AuditEventType;

typedef enum {
    AUDIT_SEV_INFO = 0,
    AUDIT_SEV_WARN = 1,
    AUDIT_SEV_ERROR = 2,
    AUDIT_SEV_CRITICAL = 3
} AuditSeverity;

#define AUDIT_MAX_DETAILS 256
#define AUDIT_MAX_EXTRA 8

typedef struct {
    uint64_t timestamp;
    uint64_t pid;
    uint32_t event_type;
    uint32_t severity;
    char details[AUDIT_MAX_DETAILS];
    uint64_t extra[AUDIT_MAX_EXTRA];
} AuditEvent;

#define AUDIT_BUF_SIZE 4096

typedef struct {
    AuditEvent buffer[AUDIT_BUF_SIZE];
    uint32_t head;
    uint32_t tail;
    uint32_t count;
    uint64_t total_events;
    uint64_t dropped_events;
    int enabled;
    int disk_log_enabled;
    uint64_t disk_log_offset;
    uint64_t disk_log_inode;
} AuditLog;

void triad_audit_init(void);
void triad_audit_log(AuditEventType type, uint32_t severity, const char *fmt, ...);
int triad_audit_get_events(AuditEvent *out, int max);
void triad_audit_flush(void);

void triad_audit_set_enabled(int enabled);
int triad_audit_is_enabled(void);

void triad_audit_set_disk_log(int enabled);
int triad_audit_get_disk_log(void);

uint64_t triad_audit_total_events(void);
uint64_t triad_audit_dropped_events(void);

void triad_audit_format_event(AuditEvent *ev, char *buf, size_t buf_size);

const char *triad_audit_event_name(AuditEventType type);
const char *triad_audit_severity_name(AuditSeverity sev);

#endif
