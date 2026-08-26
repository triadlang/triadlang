section .text
bits 64

global user_mode_init
global user_mode_enter

extern jump_user_mode

user_mode_init:
    mov rdi, [rel tss]
    call tss_init
    ret

user_mode_enter:
    mov [rel user_entry], rdi
    mov [rel user_stack_ptr], rsi
    
    mov rdi, [rel tss]
    mov rsi, [rel kernel_stack_ptr]
    call tss_set_stack
    
    cli
    mov rdi, [rel user_entry]
    mov rsi, [rel user_stack_ptr]
    call jump_user_mode
    
    sti
    ret

section .bss
align 16

user_entry:
    resq 1

user_stack_ptr:
    resq 1

kernel_stack_ptr:
    resq 1

tss:
    resb 104

section .data

gdt_user_code equ 0x18
gdt_user_data equ 0x20