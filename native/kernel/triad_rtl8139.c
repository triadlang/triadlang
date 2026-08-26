#include "triad_rtl8139.h"
#include "triad_net.h"
#include "triad_mm.h"
#include "triad_isr.h"
#include "triad_pic.h"
#include "triad_serial.h"
#include <string.h>

#define RTL_IDR        0x00
#define RTL_MAR        0x08
#define RTL_TXSTATUS0  0x10
#define RTL_TXADDR0    0x20
#define RTL_RXBUF      0x30
#define RTL_COMMAND    0x37
#define RTL_CAPR        0x38
#define RTL_IMR        0x3C
#define RTL_ISR        0x3E
#define RTL_TXCONFIG   0x40
#define RTL_RXCONFIG   0x44
#define RTL_CONFIG1    0x52
#define RTL_TIMERINT   0x54
#define RTL_PHYSTATUS  0x6E

#define RTL_CMD_RESET   0x10
#define RTL_CMD_RXEN    0x08
#define RTL_CMD_TXEN    0x04
#define RTL_CMD_BUFWRAP 0x01

#define RTL_INT_ROK    0x0001
#define RTL_INT_RER    0x0002
#define RTL_INT_TOK    0x0004
#define RTL_INT_TER    0x0008
#define RTL_INT_RXOVW  0x0010
#define RTL_INT_LINKCH 0x0020
#define RTL_INT_TIMEOUT 0x2000
#define RTL_INT_SERR   0x4000

#define RTL_RX_BUF_SIZE 8192
#define RTL_TX_BUF_SIZE 1792
#define RTL_NUM_TX_DESC 4

static uint16_t rtl_io_base = 0;
static uint8_t rtl_irq = 0;
static uint8_t rtl_mac[6] = {0};
static uint8_t *rx_buffer = NULL;
static uint8_t *tx_buffers[RTL_NUM_TX_DESC];
static int tx_cur = 0;
static int rx_pos = 0;
static bool rtl_initialized = false;

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ volatile ("outb %0, %1" : : "a"(val), "Nd"(port));
}

static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    __asm__ volatile ("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

static inline void outw(uint16_t port, uint16_t val) {
    __asm__ volatile ("outw %0, %1" : : "a"(val), "Nd"(port));
}

static inline uint16_t inw(uint16_t port) {
    uint16_t ret;
    __asm__ volatile ("inw %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

static inline void outl(uint16_t port, uint32_t val) {
    __asm__ volatile ("outl %0, %1" : : "a"(val), "Nd"(port));
}

static inline uint32_t inl(uint16_t port) {
    uint32_t ret;
    __asm__ volatile ("inl %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

void triad_rtl8139_reset(uint16_t io_base) {
    outb(io_base + RTL_COMMAND, RTL_CMD_RESET);
    for (int i = 0; i < 100; i++) {
        do { uint8_t d_; __asm__ volatile ("inb $0x80, %0" : "=a"(d_) :: "memory"); } while (0);
    }
    while (inb(io_base + RTL_COMMAND) & RTL_CMD_RESET);
}

static void rtl8139_isr_handler(TriadRegisters *regs) {
    (void)regs;
    uint16_t status = inw(rtl_io_base + RTL_ISR);

    if (status & RTL_INT_ROK) {
        outw(rtl_io_base + RTL_ISR, RTL_INT_ROK);
    }
    if (status & RTL_INT_TOK) {
        outw(rtl_io_base + RTL_ISR, RTL_INT_TOK);
    }
    if (status & RTL_INT_RXOVW) {
        outw(rtl_io_base + RTL_ISR, RTL_INT_RXOVW);
    }
    if (status & RTL_INT_LINKCH) {
        outw(rtl_io_base + RTL_ISR, RTL_INT_LINKCH);
    }

    triad_pic_eoi(rtl_irq);
}

int triad_rtl8139_init(uint16_t io_base, uint8_t irq) {
    rtl_io_base = io_base;
    rtl_irq = irq;

    triad_rtl8139_reset(io_base);

    rx_buffer = (uint8_t *)triad_mm_alloc(RTL_RX_BUF_SIZE + 16);
    if (!rx_buffer) {
        triad_serial_puts("[rtl8139] failed to allocate rx buffer\n");
        return -1;
    }
    memset(rx_buffer, 0, RTL_RX_BUF_SIZE + 16);

    for (int i = 0; i < RTL_NUM_TX_DESC; i++) {
        tx_buffers[i] = (uint8_t *)triad_mm_alloc(RTL_TX_BUF_SIZE);
        if (!tx_buffers[i]) {
            triad_serial_puts("[rtl8139] failed to allocate tx buffer\n");
            return -1;
        }
        memset(tx_buffers[i], 0, RTL_TX_BUF_SIZE);
    }

    uint32_t rx_buf_phys = (uint32_t)(uintptr_t)rx_buffer;
    outl(io_base + RTL_RXBUF, rx_buf_phys);

    for (int i = 0; i < 6; i++) {
        rtl_mac[i] = inb(io_base + RTL_IDR + i);
    }

    triad_serial_puts("[rtl8139] mac: ");
    for (int i = 0; i < 6; i++) {
        char hex[3];
        hex[0] = (rtl_mac[i] >> 4) & 0xF;
        hex[1] = rtl_mac[i] & 0xF;
        hex[2] = 0;
        triad_serial_putc(hex[0] < 10 ? '0' + hex[0] : 'A' + hex[0] - 10);
        triad_serial_putc(hex[1] < 10 ? '0' + hex[1] : 'A' + hex[1] - 10);
        if (i < 5) triad_serial_putc(':');
    }
    triad_serial_puts("\n");

    outb(io_base + RTL_CONFIG1, 0x00);

    outl(io_base + RTL_TXCONFIG, 0x03000700);
    outl(io_base + RTL_RXCONFIG, 0x00000E00);

    outw(io_base + RTL_IMR, RTL_INT_ROK | RTL_INT_TOK | RTL_INT_RXOVW | RTL_INT_LINKCH);

    outb(io_base + RTL_COMMAND, RTL_CMD_RXEN | RTL_CMD_TXEN | RTL_CMD_BUFWRAP);

    rx_pos = 0;
    tx_cur = 0;
    rtl_initialized = true;

    triad_isr_register(32 + irq, (TriadIsrCallback)rtl8139_isr_handler);
    triad_pic_unmask(irq);

    return 0;
}

int triad_rtl8139_send(const uint8_t *data, uint16_t len) {
    if (!rtl_initialized) return -1;
    if (len > 1792) len = 1792;

    uint8_t *tx_buf = tx_buffers[tx_cur];
    memcpy(tx_buf, data, len);

    outl(rtl_io_base + RTL_TXSTATUS0 + tx_cur * 4, len);

    tx_cur = (tx_cur + 1) % RTL_NUM_TX_DESC;

    return len;
}

int triad_rtl8139_recv(uint8_t *buf, uint16_t max) {
    if (!rtl_initialized) return -1;

    uint16_t status = inw(rtl_io_base + RTL_ISR);
    if (!(status & RTL_INT_ROK)) {
        return 0;
    }

    uint16_t cur_rx = inw(rtl_io_base + RTL_CAPR);
    cur_rx = (cur_rx + 0x10) % RTL_RX_BUF_SIZE;

    uint32_t header = *(uint32_t *)(rx_buffer + cur_rx);
    uint16_t rx_len = (uint16_t)(header >> 16);
    uint16_t rx_status = (uint16_t)(header & 0xFFFF);

    if (rx_status & 0x01) {
        if (rx_len > max) rx_len = max;
        uint16_t copy_len = rx_len > max ? max : rx_len;
        memcpy(buf, rx_buffer + cur_rx + 4, copy_len);

        cur_rx = (cur_rx + rx_len + 4 + 3) & ~3;
        cur_rx %= RTL_RX_BUF_SIZE;
        outw(rtl_io_base + RTL_CAPR, cur_rx - 0x10);

        outw(rtl_io_base + RTL_ISR, RTL_INT_ROK);
        return copy_len;
    }

    return 0;
}

bool triad_rtl8139_link_up(void) {
    if (!rtl_initialized) return false;
    uint8_t status = inb(rtl_io_base + RTL_PHYSTATUS);
    return (status & 0x40) != 0;
}

void triad_rtl8139_get_mac(uint8_t *mac) {
    if (!rtl_initialized) return;
    memcpy(mac, rtl_mac, 6);
}

void triad_rtl8139_handler(TriadRegisters *regs) {
    rtl8139_isr_handler(regs);
}