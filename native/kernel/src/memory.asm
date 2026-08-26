section .text
bits 64

global memory_init
global memory_get_total
global memory_get_used
global memory_get_free

memory_init:
    mov rdi, [rel mmap_addr]
    mov ecx, [rel mmap_count]
    
    xor rax, rax
    mov [rel memory_total], rax
    mov [rel memory_used], rax
    
.scan_loop:
    mov rbx, [rdi + 8]
    cmp qword [rdi + 16], 1
    jne .next_entry
    
    mov r8, [rdi]
    mov r9, rbx
    shr r9, 12
    
    add [rel memory_total], r9
    
.next_entry:
    add rdi, 24
    dec ecx
    jnz .scan_loop
    
    ret

memory_get_total:
    mov rax, [rel memory_total]
    ret

memory_get_used:
    mov rax, [rel memory_used]
    ret

memory_get_free:
    mov rax, [rel memory_total]
    sub rax, [rel memory_used]
    ret

section .bss
align 16

mmap_addr:
    resq 1

mmap_count:
    resd 1

memory_total:
    resq 1

memory_used:
    resq 1