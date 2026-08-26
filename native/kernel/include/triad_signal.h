#ifndef TRIAD_SIGNAL_H
#define TRIAD_SIGNAL_H

#include <stdint.h>
#include <stddef.h>

#define SIGKILL  9
#define SIGSEGV  11
#define SIGILL   4
#define SIGTERM  15
#define SIGINT   2
#define SIGQUIT  3
#define SIGABRT  6
#define SIGFPE   8
#define SIGPIPE  13
#define SIGALRM  14
#define SIGUSR1  10
#define SIGUSR2  12
#define SIGCHLD  17
#define SIGSTOP  19
#define SIGCONT  18
#define SIGMAX   32

typedef void (*SigHandler)(int signum);

typedef struct {
    SigHandler handlers[SIGMAX];
    uint64_t pending;
    uint64_t blocked;
    int has_pending;
} SigState;

void triad_signal_init(void);
void triad_signal_send(int pid, int signum);
void triad_signal_deliver(int pid);
int triad_signal_set_handler(int pid, int signum, SigHandler handler);
int triad_signal_block(int pid, int signum);
int triad_signal_unblock(int pid, int signum);
uint64_t triad_signal_pending(int pid);
void triad_signal_default_action(int pid, int signum);

#endif
