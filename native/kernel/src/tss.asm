section .text
bits 64

global tss_init
global tss_set_stack
global tss_get_rsp0

TSS_LIMIT equ 104

tss_init:
    mov rdi, [rel tss]
    
    xor eax, eax
    mov ecx, TSS_LIMIT / 4
.clear:
    mov dword [rdi + ecx * 4], eax
    dec ecx
    jnz .clear
    
    mov word [rdi + 0], 104
    mov qword [rdi + 4], 0
    mov qword [rdi + 8], 0
    
    mov qword [rdi + 0x04], 0
    mov qword [rdi + 0x08], 0
    
    mov qword [rdi + 24], 0
    mov qword [rdi + 32], 0
    mov qword [rdi + 40], 0
    mov qword [rdi + 48], 0
    mov qword [rdi + 56], 0
    mov qword [rdi + 64], 0
    mov qword [rdi + 72], 0
    
    mov qword [rdi + 0x14], 0
    
    ret

tss_set_stack:
    mov rdi, [rel tss]
    mov [rdi + 4], rsi
    ret

tss_get_rsp0:
    mov rax, [rel tss]
    mov rax, [rax + 4]
    ret

global jump_user_mode
jump_user_mode:
    mov rdi, [rsp + 8]
    mov rsi, [rsp + 16]
    
    mov ax, 0x23
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    
    mov rax, rdi
    push 0x23
    push rsi
    pushfq
    pop rax
    or rax, 0x200
    push rax
    push 0x1B
    push rdi
    
    mov rdx, [rel user_stack]
    push rdx
    
    mov rax, rsp
    push rax
    
    mov ax, 0x23
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    
    iretq

section .bss
align 16

tss:
    resb 104

user_stack:
    resq 1

kernel_stack:
    resq 1

section .data

tss_ptr:
    dq tss