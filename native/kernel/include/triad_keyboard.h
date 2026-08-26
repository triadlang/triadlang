#ifndef TRIAD_KEYBOARD_H
#define TRIAD_KEYBOARD_H

#include <stdint.h>
#include <stdbool.h>

#define KEYBOARD_BUFFER_SIZE 256

void triad_keyboard_init(void);
char triad_keyboard_getc(void);
int  triad_keyboard_read(char *buf, int max);
bool triad_keyboard_has_char(void);
void triad_keyboard_clear(void);

#endif