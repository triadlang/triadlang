global kmain
global g_gdt_ptr
global g_idt_ptr
global g_tss

extern kmain_c

section .boot
bits 32

start:
    cli
    mov esp, stack_top
    call check_cpuid
    call check_long_mode
    call setup_paging
    call enable_paging
    lgdt [gdt64_ptr]
    jmp 0x08:long_mode_start

check_cpuid:
    pushfd
    pop eax
    mov ecx, eax
    xor eax, 1 << 21
    push eax
    popfd
    pushfd
    pop eax
    push ecx
    popfd
    cmp eax, ecx
    je .no_cpuid
    ret
.no_cpuid:
    mov al, "1"
    jmp error

check_long_mode:
    mov eax, 0x80000000
    cpuid
    cmp eax, 0x80000001
    jb .no_long_mode
    mov eax, 0x80000001
    cpuid
    test edx, 1 << 29
    jz .no_long_mode
    ret
.no_long_mode:
    mov al, "2"
    jmp error

setup_paging:
    mov edi, 0x1000
    mov cr3, edi
    xor eax, eax
    mov ecx, 0x1000
    rep stosd
    mov edi, cr3
    
    mov DWORD [edi], 0x2003
    mov DWORD [edi + 0x800], 0x2003
    
    mov edi, 0x2000
    mov DWORD [edi], 0x0003 | (1 << 7) | (1 << 31)
    mov DWORD [edi + 8], 0x100000 | (1 << 7) | (1 << 31)
    
    mov eax, cr4
    or eax, 1 << 5
    mov cr4, eax
    ret

enable_paging:
    mov ecx, 0xC0000080
    rdmsr
    or eax, 1 << 8
    wrmsr
    mov eax, cr0
    or eax, 1 << 31
    mov cr0, eax
    ret

error:
    mov dword [0xb8000], 0x4f524f45
    mov dword [0xb8004], 0x4f3a4f52
    mov dword [0xb8008], 0x4f204f74
    mov byte  [0xb800a], al
    hlt

bits 64
section .text

long_mode_start:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    
    mov rax, higher_half_stack
    mov rsp, rax
    
    call kmain

section .data
align 4096

gdt64:
    dq 0
.code: equ $ - gdt64
    dq (1 << 43) | (1 << 44) | (1 << 47) | (1 << 53)
.data: equ $ - gdt64
    dq (1 << 44) | (1 << 47) | (1 << 41)
.tss: equ $ - gdt64
    dq 0
    dq 0
.pointer:
    dw $ - gdt64 - 1
    dq gdt64

gdt64_ptr:
    dw $ - gdt64 - 1
    dq gdt64

section .bss
align 16

stack_bottom:
    resb 65536
stack_top:

higher_half_stack:
    resb 65536

g_gdt_ptr:
    resb 10

g_idt_ptr:
    resb 2560

g_tss:
    resb 104