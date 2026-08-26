#ifndef TRIAD_RTL8139_H
#define TRIAD_RTL8139_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "triad_isr.h"

#define RTL8139_IO_BASE    0xC000
#define RTL8139_MAC_LEN    6

int triad_rtl8139_init(uint16_t io_base, uint8_t irq);
void triad_rtl8139_reset(uint16_t io_base);
int triad_rtl8139_send(const uint8_t *data, uint16_t len);
int triad_rtl8139_recv(uint8_t *buf, uint16_t max);
bool triad_rtl8139_link_up(void);
void triad_rtl8139_get_mac(uint8_t *mac);
void triad_rtl8139_handler(TriadRegisters *regs);

#endif