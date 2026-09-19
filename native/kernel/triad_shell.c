#include "triad_shell.h"
#include "triad_serial.h"
#include "triad_interpreter.h"
#include "triad_vfs.h"
#include "triad_ai.h"
#include "triad_equilibrium.h"
#include "triad_mm.h"
#include "triad_paging.h"
#include "triad_sched.h"
#include "triad_runtime_svc.h"
#include "triad_keyboard.h"
#include "triad_kernel.h"
#include <string.h>

static char shell_line[SHELL_MAX_LINE];
static int shell_line_pos = 0;
static char shell_history[SHELL_MAX_HISTORY][SHELL_MAX_LINE];
static int shell_history_count = 0;
static TriadInterp shell_interp;
static bool shell_multiline = false;
static char shell_write_path[SHELL_MAX_LINE];
static char shell_buffer[4096];
static int shell_buffer_pos = 0;

extern TriadProc triad_procs[MAX_PROCS];
extern int triad_current_pid;

static void shell_help(void) {
    triad_serial_puts("TriadLang OS shell commands:\n");
    triad_serial_puts("  .tri code         - execute TriadLang code\n");
    triad_serial_puts("  :help             - this help\n");
    triad_serial_puts("  :status           - system status\n");
    triad_serial_puts("  :procs            - list processes\n");
    triad_serial_puts("  :mem              - memory info\n");
    triad_serial_puts("  :eq               - equilibrium states\n");
    triad_serial_puts("  :ai <query>       - query native AI\n");
    triad_serial_puts("  :run <file.tri>   - run .tri from VFS\n");
    triad_serial_puts("  :ls <path>        - list VFS directory\n");
    triad_serial_puts("  :cat <path>       - print VFS file\n");
    triad_serial_puts("  :write <path>     - write to VFS file (end with .)\n");
    triad_serial_puts("  :vars             - show shell variables\n");
    triad_serial_puts("  exit              - halt system\n\n");
}

static void shell_status(void) {
    triad_serial_puts("TriadLang OS status:\n");
    triad_serial_puts("  uptime ticks: ");
    triad_serial_dec(triad_sched_get_ticks());
    triad_serial_puts("\n  threads: ");
    triad_serial_dec(triad_sched_thread_count());
    triad_serial_puts("\n  ai observations: ");
    triad_serial_dec(triad_ai_observation_count());
    triad_serial_puts("\n  memory used: ");
    triad_serial_dec(triad_mm_used());
    triad_serial_puts(" / ");
    triad_serial_dec(triad_mm_total());
    triad_serial_puts("\n  paging free: ");
    triad_serial_dec(triad_paging_free_pages());
    triad_serial_puts(" pages\n  paging used: ");
    triad_serial_dec(triad_paging_used_pages());
    triad_serial_puts(" pages\n");
}

static void shell_procs(void) {
    triad_serial_puts("processes:\n");
    for (int i = 0; i < MAX_PROCS; i++) {
        if (triad_procs[i].alive) {
            triad_serial_puts("  [");
            triad_serial_dec((uint64_t)i);
            triad_serial_puts("] ");
            triad_serial_puts(triad_procs[i].name);
            triad_serial_puts(" tier=");
            triad_serial_dec((uint64_t)triad_procs[i].tier);
            triad_serial_puts("\n");
        }
    }
}

static void shell_mem(void) {
    triad_serial_puts("memory:\n");
    triad_serial_puts("  heap used: ");
    triad_serial_dec(triad_mm_used());
    triad_serial_puts(" / ");
    triad_serial_dec(triad_mm_total());
    triad_serial_puts("\n  paging free: ");
    triad_serial_dec(triad_paging_free_pages());
    triad_serial_puts(" pages\n  paging used: ");
    triad_serial_dec(triad_paging_used_pages());
    triad_serial_puts(" pages\n");
}

static void shell_eq(void) {
    const char *names[] = {"CPU", "MEM", "NET", "GPU", "DISK", "THERM"};
    triad_serial_puts("equilibrium:\n");
    for (int i = 0; i < 6; i++) {
        double u = 0.0;
        triad_eq_read((EqHwDomain)i, &u);
        triad_serial_puts("  ");
        triad_serial_puts(names[i]);
        triad_serial_puts(": ");
        triad_serial_float(u);
        triad_serial_puts("\n");
    }
    EqState *st = triad_eq_state();
    triad_serial_puts("  fdt_precision: ");
    triad_serial_float(st->fdt_precision);
    triad_serial_puts("\n  total_energy: ");
    triad_serial_float(st->total_energy);
    triad_serial_puts("\n");
}

static void shell_ai(const char *query) {
    char buf[1024];
    int n = triad_ai_query(query, buf, sizeof(buf));
    triad_serial_puts("ai> ");
    if (n > 0) {
        triad_serial_puts(buf);
    } else {
        triad_serial_puts("no results");
    }
    triad_serial_puts("\n");
}

static void shell_ls(const char *path) {
    char files[32][256];
    int n = triad_vfs_listdir(path, files, 32);
    for (int i = 0; i < n; i++) {
        triad_serial_puts("  ");
        triad_serial_puts(files[i]);
        triad_serial_puts("\n");
    }
    if (n == 0) {
        triad_serial_puts("(empty)\n");
    }
}

static void shell_cat(const char *path) {
    int fd = triad_vfs_open(path, 0);
    if (fd < 0) {
        triad_serial_puts("file not found\n");
        return;
    }
    char buf[1024];
    int n = triad_vfs_read(fd, buf, sizeof(buf) - 1);
    triad_vfs_close(fd);
    if (n > 0) {
        buf[n] = 0;
        triad_serial_puts(buf);
        triad_serial_putc('\n');
    }
}

static void shell_run(const char *path) {
    triad_serial_puts("running: ");
    triad_serial_puts(path);
    triad_serial_puts("\n");

    int fd = triad_vfs_open(path, 0);
    if (fd < 0) {
        triad_serial_puts("file not found\n");
        return;
    }
    char buf[4096];
    int n = triad_vfs_read(fd, buf, sizeof(buf) - 1);
    triad_vfs_close(fd);
    if (n < 0) {
        triad_serial_puts("read error\n");
        return;
    }
    buf[n] = 0;

    TriadInterp *interp = (TriadInterp *)triad_mm_alloc(sizeof(TriadInterp));
    if (!interp) {
        triad_serial_puts("out of memory\n");
        return;
    }
    triad_interp_init(interp, 0);
    int rc = triad_interp_run(interp, buf);
    if (rc != 0) {
        triad_serial_puts("execution error\n");
    } else {
        triad_serial_puts("done\n");
    }
    triad_mm_free(interp);
}

static void shell_vars(void) {
    triad_serial_puts("shell variables:\n");
    for (int i = 0; i < shell_interp.scopes[shell_interp.scope_top].n_vars; i++) {
        TriadVar *v = &shell_interp.scopes[shell_interp.scope_top].vars[i];
        triad_serial_puts("  ");
        triad_serial_puts(v->name);
        triad_serial_puts(" = ");
        triad_interp_print(v->value);
        triad_serial_puts("\n");
    }
}

static void shell_write_start(const char *path) {
    int k = 0;
    while (path[k] && k < SHELL_MAX_LINE - 1) {
        shell_write_path[k] = path[k];
        k++;
    }
    shell_write_path[k] = 0;
    triad_serial_puts("writing to: ");
    triad_serial_puts(shell_write_path);
    triad_serial_puts(" (end with a line containing only '.')\n");
    shell_multiline = true;
    shell_buffer_pos = 0;
    shell_buffer[0] = 0;
}

static void shell_write_finish(void) {
    int fd = triad_vfs_open(shell_write_path, 1);
    if (fd < 0) {
        triad_serial_puts("cannot create file\n");
        return;
    }
    triad_vfs_write(fd, shell_buffer, shell_buffer_pos);
    triad_vfs_close(fd);
    triad_serial_puts("(saved)\n");
}

static int shell_handle_command(char *line) {
    if (line[0] == ':') {
        if (strncmp(line, ":help", 5) == 0) shell_help();
        else if (strncmp(line, ":status", 7) == 0) shell_status();
        else if (strncmp(line, ":procs", 6) == 0) shell_procs();
        else if (strncmp(line, ":mem", 4) == 0) shell_mem();
        else if (strncmp(line, ":eq", 3) == 0) shell_eq();
        else if (strncmp(line, ":ai ", 4) == 0) shell_ai(line + 4);
        else if (strncmp(line, ":run ", 5) == 0) shell_run(line + 5);
        else if (strncmp(line, ":ls", 3) == 0) {
            if (line[3] == ' ') shell_ls(line + 4);
            else shell_ls("/");
        }
        else if (strncmp(line, ":cat ", 5) == 0) shell_cat(line + 5);
        else if (strncmp(line, ":vars", 5) == 0) shell_vars();
        else if (strncmp(line, ":write ", 7) == 0) shell_write_start(line + 7);
        else triad_serial_puts("unknown command. try :help\n");
        return 0;
    }

    if (strcmp(line, "exit") == 0 || strcmp(line, "quit") == 0) {
        triad_serial_puts("halting system.\n");
        return -1;
    }

    triad_interp_run(&shell_interp, line);
    return 0;
}

int triad_shell_exec(const char *line) {
    return shell_handle_command((char *)line);
}

static void shell_process_char(char c) {
    if (c == '\r') c = '\n';

    if (c == '\n') {
        triad_serial_putc('\n');

        if (shell_multiline) {
            if (strcmp(shell_line, ".") == 0) {
                shell_multiline = false;
                shell_write_finish();
                shell_line_pos = 0;
                shell_line[0] = 0;
                triad_shell_prompt();
                return;
            }
            int k = 0;
            while (shell_line[k] && shell_buffer_pos < (int)sizeof(shell_buffer) - 2) {
                shell_buffer[shell_buffer_pos] = shell_line[k];
                shell_buffer_pos++; k++;
            }
            shell_buffer[shell_buffer_pos] = '\n';
            shell_buffer_pos++;
            shell_buffer[shell_buffer_pos] = 0;
        } else {
            shell_line[shell_line_pos] = 0;
            if (shell_line_pos > 0) {
                int k = 0;
                while (shell_line[k] && k < SHELL_MAX_LINE - 1) {
                    shell_history[shell_history_count % SHELL_MAX_HISTORY][k] = shell_line[k];
                    k++;
                }
                shell_history[shell_history_count % SHELL_MAX_HISTORY][k] = 0;
                shell_history_count++;

                if (shell_handle_command(shell_line) < 0) {
                    shell_line_pos = 0;
                    shell_line[0] = 0;
                    return;
                }
            }
        }
        shell_line_pos = 0;
        shell_line[0] = 0;
        triad_shell_prompt();
        return;
    }

    if (c == 8 || c == 127) {
        if (shell_line_pos > 0) {
            shell_line_pos--;
            triad_serial_puts("\b \b");
        }
        return;
    }

    if (shell_line_pos < SHELL_MAX_LINE - 1) {
        shell_line[shell_line_pos] = c;
        shell_line_pos++;
        triad_serial_putc(c);
    }
}

void triad_shell_init(void) {
    shell_line_pos = 0;
    shell_line[0] = 0;
    shell_history_count = 0;
    shell_multiline = false;
    shell_buffer_pos = 0;
    shell_buffer[0] = 0;
    triad_interp_init(&shell_interp, 0);
}

void triad_shell_prompt(void) {
    if (shell_multiline) {
        triad_serial_puts("... ");
    } else {
        triad_serial_puts("triad> ");
    }
}

void triad_shell_run(void) {
    triad_serial_puts("\nTriadLang OS Shell v1.0\n");
    triad_serial_puts("Type :help for commands, or enter .tri code directly.\n\n");
    triad_shell_prompt();

    while (1) {
        char c = 0;

        if (triad_keyboard_has_char()) {
            c = triad_keyboard_getc();
        } else {
            c = triad_serial_getc();
        }

        if (c) {
            shell_process_char(c);
        }
    }
}