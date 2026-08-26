#include "triad_keyboard.h"
#include "triad_isr.h"
#include "triad_pic.h"

static char keyboard_buffer[KEYBOARD_BUFFER_SIZE];
static int  keyboard_head = 0;
static int  keyboard_tail = 0;
static bool keyboard_shift = false;
static bool keyboard_ctrl = false;
static bool keyboard_alt = false;
static bool keyboard_caps_lock = false;
static bool keyboard_num_lock = false;

static const char scancode_to_ascii[128] = {
    0,    0,   '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', '\b',
    '\t', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\n',
    0,    'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\'', '`', 0,
    '\\', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 0, '*', 0, ' ', 0,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};

static const char scancode_to_ascii_shift[128] = {
    0,    0,   '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '\b',
    '\t', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', '\n',
    0,    'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"', '~', 0,
    '|',  'Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?', 0, '*', 0, ' ', 0,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};

static const char scancode_f_keys[12] = {
    '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08', '\x09', '\x0A', '\x0B', '\x0C'
};

static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    __asm__ volatile ("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ volatile ("outb %0, %1" : : "a"(val), "Nd"(port));
}

static void keyboard_handler(TriadRegisters *regs) {
    (void)regs;
    
    uint8_t scancode = inb(0x60);
    uint8_t code = scancode & 0x7F;
    bool release = (scancode & 0x80) != 0;
    
    if (release) {
        switch (code) {
            case 0x2A:
            case 0x36:
                keyboard_shift = false;
                break;
            case 0x1D:
                keyboard_ctrl = false;
                break;
            case 0x38:
                keyboard_alt = false;
                break;
        }
        return;
    }
    
    switch (code) {
        case 0x2A:
        case 0x36:
            keyboard_shift = true;
            return;
        case 0x1D:
            keyboard_ctrl = true;
            return;
        case 0x38:
            keyboard_alt = true;
            return;
        case 0x3A:
            keyboard_caps_lock = !keyboard_caps_lock;
            return;
        case 0x45:
            keyboard_num_lock = !keyboard_num_lock;
            return;
    }
    
    if (code >= 0x3B && code <= 0x46) {
        int fkey = code - 0x3B;
        if (fkey < 12) {
            int next = (keyboard_head + 1) % KEYBOARD_BUFFER_SIZE;
            if (next != keyboard_tail) {
                keyboard_buffer[keyboard_head] = scancode_f_keys[fkey];
                keyboard_head = next;
            }
        }
        return;
    }
    
    char c;
    if (keyboard_shift) {
        c = scancode_to_ascii_shift[code];
    } else {
        c = scancode_to_ascii[code];
    }
    
    if (keyboard_caps_lock && c >= 'a' && c <= 'z') {
        c = c - 'a' + 'A';
    } else if (keyboard_caps_lock && c >= 'A' && c <= 'Z') {
        c = c - 'A' + 'a';
    }
    
    if (c) {
        int next = (keyboard_head + 1) % KEYBOARD_BUFFER_SIZE;
        if (next != keyboard_tail) {
            keyboard_buffer[keyboard_head] = c;
            keyboard_head = next;
        }
    }
}

void triad_keyboard_init(void) {
    keyboard_head = 0;
    keyboard_tail = 0;
    keyboard_shift = false;
    keyboard_ctrl = false;
    keyboard_alt = false;
    keyboard_caps_lock = false;
    keyboard_num_lock = false;
    
    triad_isr_register(IRQ_KEYBOARD, keyboard_handler);
    triad_pic_unmask(1);
}

char triad_keyboard_getc(void) {
    if (keyboard_head == keyboard_tail) {
        return 0;
    }
    char c = keyboard_buffer[keyboard_tail];
    keyboard_tail = (keyboard_tail + 1) % KEYBOARD_BUFFER_SIZE;
    return c;
}

int triad_keyboard_read(char *buf, int max) {
    int count = 0;
    while (count < max && keyboard_head != keyboard_tail) {
        buf[count] = keyboard_buffer[keyboard_tail];
        keyboard_tail = (keyboard_tail + 1) % KEYBOARD_BUFFER_SIZE;
        count++;
    }
    return count;
}

bool triad_keyboard_has_char(void) {
    return keyboard_head != keyboard_tail;
}

void triad_keyboard_clear(void) {
    keyboard_head = 0;
    keyboard_tail = 0;
}