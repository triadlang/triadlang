#include "triad_signal.h"
#include "triad_kernel.h"
#include "triad_sched.h"
#include "triad_serial.h"
#include "triad_audit.h"
#include <string.h>

extern TriadProc triad_procs[MAX_PROCS];

static SigState signal_states[MAX_PROCS];

void triad_signal_init(void) {
    for (int i = 0; i < MAX_PROCS; i++) {
        memset(&signal_states[i], 0, sizeof(SigState));
        signal_states[i].handlers[SIGKILL] = NULL;
        signal_states[i].handlers[SIGSEGV] = NULL;
        signal_states[i].handlers[SIGILL] = NULL;
        signal_states[i].handlers[SIGTERM] = NULL;
        signal_states[i].handlers[SIGINT] = NULL;
        signal_states[i].handlers[SIGQUIT] = NULL;
        signal_states[i].handlers[SIGABRT] = NULL;
        signal_states[i].handlers[SIGFPE] = NULL;
        signal_states[i].handlers[SIGPIPE] = NULL;
        signal_states[i].handlers[SIGALRM] = NULL;
        signal_states[i].handlers[SIGUSR1] = NULL;
        signal_states[i].handlers[SIGUSR2] = NULL;
        signal_states[i].handlers[SIGCHLD] = NULL;
        signal_states[i].handlers[SIGSTOP] = NULL;
        signal_states[i].handlers[SIGCONT] = NULL;
        signal_states[i].pending = 0;
        signal_states[i].blocked = 0;
        signal_states[i].has_pending = 0;
    }
}

void triad_signal_send(int pid, int signum) {
    if (pid < 0 || pid >= MAX_PROCS) return;
    if (signum < 0 || signum >= SIGMAX) return;

    if (!triad_procs[pid].alive) return;

    if (signal_states[pid].blocked & (1ULL << signum)) {
        signal_states[pid].pending |= (1ULL << signum);
        signal_states[pid].has_pending = 1;
        return;
    }

    if (signum == SIGKILL || signum == SIGSTOP) {
        triad_signal_default_action(pid, signum);
        return;
    }

    SigHandler handler = signal_states[pid].handlers[signum];
    if (handler) {
        handler(signum);
    } else {
        triad_signal_default_action(pid, signum);
    }
}

void triad_signal_deliver(int pid) {
    if (pid < 0 || pid >= MAX_PROCS) return;
    if (!signal_states[pid].has_pending) return;

    uint64_t pending = signal_states[pid].pending;
    signal_states[pid].pending = 0;
    signal_states[pid].has_pending = 0;

    for (int sig = 0; sig < SIGMAX; sig++) {
        if (pending & (1ULL << sig)) {
            if (signal_states[pid].blocked & (1ULL << sig)) {
                signal_states[pid].pending |= (1ULL << sig);
                signal_states[pid].has_pending = 1;
                continue;
            }
            triad_signal_send(pid, sig);
        }
    }
}

int triad_signal_set_handler(int pid, int signum, SigHandler handler) {
    if (pid < 0 || pid >= MAX_PROCS) return -1;
    if (signum < 0 || signum >= SIGMAX) return -1;
    if (signum == SIGKILL || signum == SIGSTOP) return -1;

    signal_states[pid].handlers[signum] = handler;
    return 0;
}

int triad_signal_block(int pid, int signum) {
    if (pid < 0 || pid >= MAX_PROCS) return -1;
    if (signum < 0 || signum >= SIGMAX) return -1;
    if (signum == SIGKILL || signum == SIGSTOP) return -1;

    signal_states[pid].blocked |= (1ULL << signum);
    return 0;
}

int triad_signal_unblock(int pid, int signum) {
    if (pid < 0 || pid >= MAX_PROCS) return -1;
    if (signum < 0 || signum >= SIGMAX) return -1;

    signal_states[pid].blocked &= ~(1ULL << signum);

    if (signal_states[pid].has_pending && (signal_states[pid].pending & (1ULL << signum))) {
        triad_signal_deliver(pid);
    }

    return 0;
}

uint64_t triad_signal_pending(int pid) {
    if (pid < 0 || pid >= MAX_PROCS) return 0;
    return signal_states[pid].pending;
}

void triad_signal_default_action(int pid, int signum) {
    switch (signum) {
    case SIGKILL:
    case SIGTERM:
    case SIGQUIT:
    case SIGINT:
        triad_audit_log(AUDIT_PROC_KILL, AUDIT_SEV_INFO,
                        "process %d killed by signal %d", pid, signum);
        triad_procs[pid].alive = false;
        break;
    case SIGSEGV:
        triad_audit_log(AUDIT_PAGE_FAULT, AUDIT_SEV_ERROR,
                        "process %d segmentation fault", pid);
        triad_procs[pid].alive = false;
        break;
    case SIGILL:
        triad_audit_log(AUDIT_EXEC_SIG_FAIL, AUDIT_SEV_ERROR,
                        "process %d illegal instruction", pid);
        triad_procs[pid].alive = false;
        break;
    case SIGFPE:
        triad_audit_log(AUDIT_EXEC_SIG_FAIL, AUDIT_SEV_ERROR,
                        "process %d floating point exception", pid);
        triad_procs[pid].alive = false;
        break;
    case SIGABRT:
        triad_audit_log(AUDIT_PROC_KILL, AUDIT_SEV_ERROR,
                        "process %d aborted", pid);
        triad_procs[pid].alive = false;
        break;
    case SIGSTOP:
        break;
    case SIGCONT:
        break;
    case SIGPIPE:
        break;
    case SIGCHLD:
        break;
    default:
        triad_audit_log(AUDIT_PROC_KILL, AUDIT_SEV_WARN,
                        "process %d killed by signal %d", pid, signum);
        triad_procs[pid].alive = false;
        break;
    }
}
