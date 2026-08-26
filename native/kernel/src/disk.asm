section .text
bits 64

global disk_init
global disk_read
global disk_write
global disk_get_size

DISK_SECTOR_SIZE equ 512
DISK_CACHE_SIZE equ 64

disk_init:
    call ata_init
    
    mov qword [rel disk_cache_head], 0
    mov qword [rel disk_cache_tail], 0
    
    mov rdi, disk_cache
    mov ecx, DISK_CACHE_SIZE
.init_cache:
    mov qword [rdi], 0
    mov qword [rdi + 8], 0
    add rdi, 16
    dec ecx
    jnz .init_cache
    
    ret

disk_read:
    push rbx
    push r12
    push r13
    push r14
    
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    
    mov rdi, r12
    call disk_cache_lookup
    test rax, rax
    jz .cache_miss
    
    mov rdi, rax
    mov rsi, r14
    mov ecx, 512
    rep movsb
    
    mov rax, 512
    jmp .done
    
.cache_miss:
    mov rax, r12
    shr rax, 9
    
    mov rdi, rax
    mov rsi, r14
    call ata_read_sector
    
    mov rax, 512
    
.done:
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

disk_write:
    push rbx
    push r12
    push r13
    
    mov r12, rdi
    mov r13, rsi
    
    mov rax, r12
    shr rax, 9
    
    mov rdi, rax
    mov rsi, r13
    call ata_write_sector
    
    call disk_cache_invalidate
    
    mov rax, 512
    
    pop r13
    pop r12
    pop rbx
    ret

disk_get_size:
    mov rax, [rel disk_total_sectors]
    shl rax, 9
    ret

disk_cache_lookup:
    mov rax, [rel disk_cache_head]
    
.loop:
    cmp rax, [rel disk_cache_tail]
    je .not_found
    
    mov rbx, disk_cache
    shl rax, 4
    add rbx, rax
    
    mov rcx, [rbx]
    cmp rcx, rdi
    je .found
    
    inc rax
    and rax, DISK_CACHE_SIZE - 1
    jmp .loop
    
.found:
    mov rax, rbx
    add rax, 8
    ret
    
.not_found:
    xor eax, eax
    ret

disk_cache_invalidate:
    mov rax, [rel disk_cache_head]
    
.loop:
    cmp rax, [rel disk_cache_tail]
    je .done
    
    mov rbx, disk_cache
    shl rax, 4
    add rbx, rax
    
    mov rcx, [rbx]
    shr rcx, 9
    cmp rcx, rdi
    jne .next
    
    mov qword [rbx], 0
    
.next:
    inc rax
    and rax, DISK_CACHE_SIZE - 1
    jmp .loop
    
.done:
    ret

section .bss
align 16

disk_cache:
    resb DISK_CACHE_SIZE * 16

disk_cache_head:
    resq 1

disk_cache_tail:
    resq 1

disk_total_sectors:
    resq 1