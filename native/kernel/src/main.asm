section .text
bits 64

global kmain
global kmain_c
global isr_dispatcher

extern gdt_init
extern idt_init
extern pic_init
extern pit_init
extern vga_init
extern memory_init
extern paging_init
extern heap_init
extern tss_init
extern ata_init
extern vfs_init
extern keyboard_init
extern scheduler_init
extern elf_load
extern triad_kernel_init
extern qos_init
extern qos_llm_init
extern qos_kernel_init
extern qos_scheduler_init
extern qos_shell_start

section .text

kmain:
    cli
    
    mov rdi, [rel mmap_addr]
    mov esi, [rel mmap_count]
    call memory_init
    
    call gdt_init
    call idt_init
    call pic_init
    call pit_init
    call vga_init
    call paging_init
    call heap_init
    call tss_init
    call ata_init
    call vfs_init
    call keyboard_init
    
    call scheduler_init
    call triad_kernel_init
    
    call qos_kernel_init
    call qos_scheduler_init
    
    sti
    
    call qos_shell_start
    
    jmp .halt

.halt:
    hlt
    jmp .halt

kmain_c:
    ret

isr_dispatcher:
    push rdi
    push rsi
    
    mov rdi, [rsp + 16]
    
    cmp qword [rsp + 24], 0x80
    je .syscall
    
    cmp qword [rsp + 24], 14
    je .page_fault
    
    cmp qword [rsp + 24], 13
    je .gpf
    
    cmp qword [rsp + 24], 8
    je .double_fault
    
    jmp .unknown

.syscall:
    call syscall_handler
    jmp .done

.page_fault:
    mov rdi, .pf_msg
    call vga_puts
    jmp .halt

.gpf:
    mov rdi, .gpf_msg
    call vga_puts
    jmp .halt

.double_fault:
    mov rdi, .df_msg
    call vga_puts
    jmp .halt

.unknown:
    mov rdi, .unk_msg
    call vga_puts
    
.halt:
    pop rsi
    pop rdi
    
    add rsp, 16
    iretq
    
.done:
    pop rsi
    pop rdi
    add rsp, 16
    iretq

shell_start:
    mov rdi, .welcome_msg
    call vga_puts
    
.shell_loop:
    mov rdi, .prompt_msg
    call vga_puts
    
.wait_input:
    call keyboard_has_key
    test al, al
    jz .wait_input
    
    call keyboard_get_key
    test al, al
    jz .wait_input
    
    cmp al, 13
    je .execute_cmd
    
    push rdi
    movzx rdi, al
    call vga_putc
    pop rdi
    
    jmp .wait_input
    
.execute_cmd:
    call vga_putc
    
    mov rdi, .newline_msg
    call vga_puts
    
    jmp .shell_loop
    
    ret

section .rodata

welcome_msg:
    db "TriadOS v0.1 - TriadLang Kernel", 13, 10
    db "=================================", 13, 10, 0

prompt_msg:
    db "triad> ", 0

newline_msg:
    db 13, 10, 0

pf_msg:
    db "Page fault!", 13, 10, 0

gpf_msg:
    db "General protection fault!", 13, 10, 0

df_msg:
    db "Double fault!", 13, 10, 0

unk_msg:
    db "Unknown exception!", 13, 10, 0

section .bss
align 16

mmap_addr:
    resq 1

mmap_count:
    resd 1