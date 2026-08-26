section .text
bits 64

global elf_load
global elf_check_header
global elf_get_entry
global elf_load_segments

ELF_MAGIC equ 0x464C457F
ELF_CLASS_64 equ 2
ELF_DATA_LE equ 1
ELF_TYPE_EXEC equ 2
ELF_TYPE_DYN equ 3

PT_NULL equ 0
PT_LOAD equ 1
PT_DYNAMIC equ 2
PT_INTERP equ 3
PT_NOTE equ 4
PT_SHLIB equ 5
PT_PHDR equ 6

PF_X equ 1
PF_W equ 2
PF_R equ 4

elf_check_header:
    push rbx
    
    mov eax, [rdi]
    cmp eax, ELF_MAGIC
    jne .fail
    
    mov al, [rdi + 4]
    cmp al, ELF_CLASS_64
    jne .fail
    
    mov al, [rdi + 5]
    cmp al, ELF_DATA_LE
    jne .fail
    
    movzx eax, word [rdi + 16]
    cmp ax, ELF_TYPE_EXEC
    je .ok
    cmp ax, ELF_TYPE_DYN
    jne .fail
    
.ok:
    mov al, [rdi + 6]
    cmp al, 1
    jne .fail
    
    mov al, 1
    pop rbx
    ret
    
.fail:
    xor eax, eax
    pop rbx
    ret

elf_get_entry:
    mov rax, [rdi + 24]
    ret

elf_load:
    push rbx
    push r12
    push r13
    push r14
    
    mov r12, rdi
    mov r13, rsi
    
    call elf_check_header
    test al, al
    jz .fail
    
    mov rax, [r12 + 60]
    movzx ecx, word [r12 + 54]
    
    mov rbx, r12
    add rbx, rax
    
    movzx edx, word [r12 + 56]
    mov r14d, edx
    
.load_segment:
    test edx, edx
    jz .done
    
    mov eax, [rbx]
    cmp eax, PT_LOAD
    jne .next_segment
    
    mov rdi, [rbx + 16]
    
    mov rsi, [rbx + 8]
    add rsi, r12
    
    mov rcx, [rbx + 40]
    
    push rbx
    push rcx
    
    call elf_load_segment
    
    pop rcx
    pop rbx
    
.next_segment:
    add rbx, ecx
    dec edx
    jmp .load_segment
    
.done:
    mov rax, [r12 + 24]
    
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
    
.fail:
    xor eax, eax
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

elf_load_segment:
    push rbx
    push r12
    push r13
    
    mov r12, rdi
    mov r13, rsi
    
    mov rax, r12
    and rax, 0xFFF
    mov r12, rdi
    sub r12, rax
    
    mov rdi, r12
    mov rsi, [rbx + 32]
    
    call paging_map
    
    push rdi
    push rsi
    
    mov rdi, r12
    shr rdi, 9
    
    mov rsi, r13
    mov ecx, [rbx + 40]
    rep movsb
    
    mov ecx, [rbx + 48]
    sub ecx, [rbx + 40]
    jle .no_bss
    mov al, 0
    rep stosb
    
.no_bss:
    pop rsi
    pop rdi
    
    pop r13
    pop r12
    pop rbx
    ret

elf_get_phdr:
    mov rax, [rdi + 60]
    add rax, rdi
    ret

elf_get_shdr:
    mov rax, [rdi + 40]
    add rax, rdi
    ret

elf_find_section:
    push rbx
    push r12
    
    mov r12, rdi
    mov rdi, [r12 + 40]
    add rdi, r12
    
    movzx ecx, word [r12 + 58]
    movzx eax, word [r12 + 62]
    
    mov rbx, rsi
    
.find_loop:
    test ecx, ecx
    jz .not_found
    
    mov edx, [rdi + 4]
    cmp edx, ebx
    je .found
    
    add rdi, rax
    dec ecx
    jmp .find_loop
    
.found:
    mov rax, [rdi + 24]
    add rax, r12
    jmp .done
    
.not_found:
    xor eax, eax
    
.done:
    pop r12
    pop rbx
    ret

section .bss
align 16

elf_buffer:
    resb 4096