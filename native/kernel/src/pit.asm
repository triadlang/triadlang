section .text
bits 64

global pit_init
global pit_sleep_ms
global pit_get_ticks

pit_init:
    mov al, 0x36
    out 0x43, al
    
    mov ax, 11931
    out 0x40, al
    mov al, ah
    out 0x40, al
    
    ret

pit_get_ticks:
    mov rax, [rel pit_ticks]
    ret

pit_sleep_ms:
    push rbx
    mov rbx, rdi
    
    mov rax, [rel pit_ticks]
    add rax, rbx
    
.wait_loop:
    hlt
    mov rcx, [rel pit_ticks]
    cmp rcx, rax
    jl .wait_loop
    
    pop rbx
    ret

global pit_handler
pit_handler:
    push rax
    push rbx
    push rcx
    push rdx
    
    inc qword [rel pit_ticks]
    
    mov al, 0x20
    out 0x20, al
    
    pop rdx
    pop rcx
    pop rbx
    pop rax
    iretq

section .bss
align 8

pit_ticks:
    resq 1