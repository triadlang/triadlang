section .text
bits 64

global idt_init
global idt_set_gate

extern isr_handler_table

idt_init:
    mov rdi, idt64
    mov rcx, 256
    xor rax, rax
.loop:
    mov qword [rdi], rax
    mov qword [rdi + 8], rax
    add rdi, 16
    dec rcx
    jnz .loop
    
    mov rdi, 0
    mov rsi, isr_divide_error
    call idt_set_gate
    
    mov rdi, 1
    mov rsi, isr_debug
    call idt_set_gate
    
    mov rdi, 2
    mov rsi, isr_nmi
    call idt_set_gate
    
    mov rdi, 3
    mov rsi, isr_breakpoint
    call idt_set_gate
    
    mov rdi, 6
    mov rsi, isr_invalid_opcode
    call idt_set_gate
    
    mov rdi, 8
    mov rsi, isr_double_fault
    call idt_set_gate
    
    mov rdi, 13
    mov rsi, isr_general_protection
    call idt_set_gate
    
    mov rdi, 14
    mov rsi, isr_page_fault
    call idt_set_gate
    
    mov rdi, 0x80
    mov rsi, isr_syscall
    call idt_set_gate
    
    lidt [rel idt64_ptr]
    ret

idt_set_gate:
    mov r8, rdi
    mov rax, rsi
    
    lea rdi, [rel idt64]
    shl r8, 4
    add rdi, r8
    
    mov word [rdi], ax
    mov word [rdi + 2], 0x08
    mov byte [rdi + 4], 0
    mov byte [rdi + 5], 0x8E
    shr rax, 16
    mov word [rdi + 6], ax
    shr rax, 16
    mov dword [rdi + 8], eax
    mov dword [rdi + 12], 0
    ret

section .data
align 16

idt64:
    resb 4096

idt64_ptr:
    dw 4095
    dq idt64

section .text

isr_divide_error:
    push 0
    push 0
    jmp isr_common

isr_debug:
    push 0
    push 1
    jmp isr_common

isr_nmi:
    push 0
    push 2
    jmp isr_common

isr_breakpoint:
    push 0
    push 3
    jmp isr_common

isr_invalid_opcode:
    push 0
    push 6
    jmp isr_common

isr_double_fault:
    push 8
    jmp isr_common

isr_general_protection:
    push 13
    jmp isr_common

isr_page_fault:
    push 14
    jmp isr_common

isr_syscall:
    push 0x80
    jmp isr_common

isr_common:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    
    mov rdi, rsp
    call isr_dispatcher
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    pop rbp
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    
    add rsp, 16
    iretq