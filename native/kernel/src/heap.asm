section .text
bits 64

global heap_init
global heap_alloc
global heap_free
global heap_realloc
global heap_stats

HEAP_START equ 0xFFFFFFFF80010000
HEAP_SIZE equ 0x10000000
BLOCK_SIZE equ 16

heap_init:
    mov qword [rel heap_start], HEAP_START
    mov qword [rel heap_end], HEAP_START + HEAP_SIZE
    mov qword [rel heap_used], 0
    mov qword [rel heap_blocks], HEAP_SIZE / BLOCK_SIZE
    
    mov rdi, HEAP_START
    mov rax, 0
    mov ecx, BLOCK_SIZE / 8
.clear_block:
    mov qword [rdi], rax
    add rdi, 8
    loop .clear_block
    
    mov rax, [rel heap_end]
    sub rax, [rel heap_start]
    sub rax, 8
    
    mov rdi, [rel heap_start]
    mov qword [rdi], rax
    mov qword [rdi + 8], 0
    
    ret

heap_alloc:
    push rbx
    push r12
    push r13
    
    mov r12, rdi
    add r12, 15
    and r12, ~15
    
    mov r13, [rel heap_start]
    
.find_block:
    mov rbx, [r13]
    test rbx, rbx
    jz .not_found
    
    mov eax, ebx
    and eax, 0x1
    
    test eax, eax
    jnz .next_block
    
    mov rcx, [r13]
    cmp rcx, r12
    jb .next_block
    
    mov rax, [r13]
    sub rax, r12
    cmp rax, 16
    jge .split_block
    
    mov rax, [r13]
    or rax, 1
    mov [r13], rax
    
    lea rax, [r13 + 8]
    add qword [rel heap_used], r12
    jmp .done
    
.split_block:
    push rdi
    mov rdi, r13
    add rdi, 8
    add rdi, r12
    add rdi, 8
    
    mov [rdi], rax
    sub rax, 8
    mov [rdi], rax
    
    mov rax, r12
    or rax, 1
    mov [r13], rax
    
    pop rdi
    
    lea rax, [r13 + 8]
    add qword [rel heap_used], r12
    jmp .done
    
.next_block:
    mov rcx, [r13]
    and rcx, ~1
    lea r13, [r13 + 8 + rcx]
    jmp .find_block
    
.not_found:
    xor eax, eax
    
.done:
    pop r13
    pop r12
    pop rbx
    ret

heap_free:
    push rbx
    
    test rdi, rdi
    jz .done
    
    mov rbx, rdi
    sub rbx, 8
    
    mov rax, [rbx]
    and rax, ~1
    mov [rbx], rax
    
    mov rax, [rbx]
    shr rax, 3
    sub qword [rel heap_used], rax
    
    mov rcx, rbx
    sub rcx, 8
    
    test rcx, rcx
    js .done
    
    mov rdx, [rcx]
    test rdx, rdx
    js .done
    test rdx, 1
    jnz .done
    
    mov rax, [rcx]
    add rax, 16
    add rax, [rbx]
    mov [rcx], rax
    jmp .done
    
.done:
    pop rbx
    ret

heap_realloc:
    push rbx
    push r12
    
    test rdi, rdi
    jz .alloc_new
    
    mov rbx, rdi
    mov r12, rsi
    
    mov rdi, r12
    call heap_alloc
    test rax, rax
    jz .fail
    
    push rax
    mov rdi, rbx
    mov rsi, rax
    call heap_size
    
    mov rdx, rax
    cmp rdx, r12
    jle .copy_size_ok
    mov rdx, r12
    
.copy_size_ok:
    mov rdi, [rsp]
    mov rsi, rbx
    mov rcx, rdx
    rep movsb
    
    mov rdi, rbx
    call heap_free
    
    pop rax
    jmp .done
    
.alloc_new:
    mov rdi, rsi
    call heap_alloc
    jmp .done
    
.fail:
    xor eax, eax
    
.done:
    pop r12
    pop rbx
    ret

heap_stats:
    mov rax, [rel heap_used]
    mov rdx, HEAP_SIZE
    sub rdx, [rel heap_used]
    ret

heap_size:
    test rdi, rdi
    jz .fail
    mov rax, [rdi - 8]
    and rax, ~1
    ret
.fail:
    xor eax, eax
    ret

section .bss
align 16

heap_start:
    resq 1

heap_end:
    resq 1

heap_used:
    resq 1

heap_blocks:
    resq 1