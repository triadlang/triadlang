section .text
bits 64

global gdt_init
global gdt_load_tss

gdt_init:
    lgdt [rel gdt64_ptr]
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    ret

gdt_load_tss:
    mov ax, 0x28
    ltr ax
    ret

section .data
align 16

gdt64:
    dq 0
.kernel_code: equ $ - gdt64
    dq 0x00AF9A000000FFFF
.kernel_data: equ $ - gdt64
    dq 0x00AF92000000FFFF
.user_code: equ $ - gdt64
    dq 0x00AFFA000000FFFF
.user_data: equ $ - gdt64
    dq 0x00AFF2000000FFFF
.tss_segment: equ $ - gdt64
    dq 0
    dq 0
gdt64_end:

gdt64_ptr:
    dw gdt64_end - gdt64 - 1
    dq gdt64

section .bss
align 4096

tss_struct:
    resb 104