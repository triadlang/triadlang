#ifndef TRIAD_SHELL_H
#define TRIAD_SHELL_H

#include "triad_kernel.h"

void triad_shell_init(void);
void triad_shell_run(void);
void triad_shell_prompt(void);
int  triad_shell_exec(const char *line);

#define SHELL_MAX_LINE 512
#define SHELL_MAX_HISTORY 64

#endif