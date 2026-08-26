section .text
bits 64

global paging_init
global paging_map
global paging_unmap
global paging_get_physical

PAGE_SIZE equ 4096
PAGE_PRESENT equ 0x01
PAGE_WRITABLE equ 0x02
PAGE_USER equ 0x04
PAGE_HUGE equ 0x80

paging_init:
    mov rdi, [rel pml4]
    mov cr3, rdi
    ret

paging_map:
    mov rax, rdi
    mov rdi, rsi
    mov rsi, rdx
    
    mov rcx, rdi
    and rcx, 0x1FF
    shr rdi, 9
    and rdi, 0x1FF
    shr rdi, 9
    and rdi, 0x1FF
    shr rdi, 9
    
    mov rbx, [rel pml4]
    lea rbx, [rbx + rdi * 8]
    
    test qword [rbx], 1
    jnz .pdpt_exists
    
    call alloc_page
    or rax, PAGE_PRESENT | PAGE_WRITABLE | PAGE_USER
    mov [rbx], rax
    
.pdpt_exists:
    mov rbx, [rbx]
    and rbx, ~0xFFF
    lea rbx, [rbx + rdi * 8]
    
    test qword [rbx], 1
    jnz .pd_exists
    
    call alloc_page
    or rax, PAGE_PRESENT | PAGE_WRITABLE | PAGE_USER
    mov [rbx], rax
    
.pd_exists:
    mov rbx, [rbx]
    and rbx, ~0xFFF
    lea rbx, [rbx + rdi * 8]
    
    test qword [rbx], 1
    jnz .pt_exists
    
    call alloc_page
    or rax, PAGE_PRESENT | PAGE_WRITABLE | PAGE_USER
    mov [rbx], rax
    
.pt_exists:
    mov rbx, [rbx]
    and rbx, ~0xFFF
    lea rbx, [rbx + rcx * 8]
    
    mov rax, rsi
    or rax, PAGE_PRESENT | PAGE_WRITABLE | PAGE_USER
    mov [rbx], rax
    
    ret

paging_unmap:
    mov rcx, rdi
    and rcx, 0x1FF
    
    shr rdi, 9
    and rdi, 0x1FF
    shr rdi, 9
    and rdi, 0x1FF
    shr rdi, 9
    
    mov rbx, [rel pml4]
    lea rbx, [rbx + rdi * 8]
    
    test qword [rbx], 1
    jz .done
    
    mov rbx, [rbx]
    and rbx, ~0xFFF
    lea rbx, [rbx + rdi * 8]
    
    shr rdi, 9
    test qword [rbx], 1
    jz .done
    
    mov rbx, [rbx]
    and rbx, ~0xFFF
    lea rbx, [rbx + rcx * 8]
    
    mov qword [rbx], 0
    invlpg [rdi]
    
.done:
    ret

paging_get_physical:
    mov rax, rdi
    shr rax, 9
    and rax, 0x1FF
    
    mov rbx, [rel pml4]
    mov rbx, [rbx + rax * 8]
    and rbx, ~0xFFF
    
    mov rcx, rdi
    shr rcx, 18
    and rcx, 0x1FF
    
    mov rbx, [rbx + rcx * 8]
    and rbx, ~0xFFF
    
    shr rdi, 27
    and rdi, 0x1FF
    
    mov rbx, [rbx + rdi * 8]
    and rbx, ~0xFFF
    
    and rdi, 0x1FF
    
    mov rax, [rbx + rdi * 8]
    and rax, ~0xFFF
    
    mov rcx, rax
    and rcx, 0xFFF
    or rax, rcx
    
    ret

alloc_page:
    push rbx
    mov rbx, [rel next_free_page]
    test rbx, rbx
    jz .out_of_memory
    
    mov rax, rbx
    mov rbx, [rbx]
    mov [rel next_free_page], rbx
    
    push rdi
    push rcx
    mov rdi, rax
    xor eax, eax
    mov ecx, PAGE_SIZE / 8
    rep stosq
    pop rcx
    pop rdi
    
    pop rbx
    ret
    
.out_of_memory:
    xor eax, eax
    pop rbx
    ret

section .bss
align 4096

pml4:
    resb PAGE_SIZE

pdpt:
    resb PAGE_SIZE

pd:
    resb PAGE_SIZE

pt:
    resb PAGE_SIZE * 512

next_free_page:
    resq 1