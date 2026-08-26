.set MB_MAGIC,    0x1BADB002
.set MB_FLAGS,    0x00000003
.set MB_CHECKSUM, -(MB_MAGIC + MB_FLAGS)

.section .multiboot,"a",@progbits
.align 4
    .long MB_MAGIC
    .long MB_FLAGS
    .long MB_CHECKSUM

.section .bss
.align 16
stack_bottom:
    .skip 16384
stack_top:

.align 4096
pml4:
    .skip 4096
pdpt:
    .skip 4096
pd:
    .skip 4096

.section .rodata
.align 8
gdt64:
    .quad 0x0000000000000000   # null
    .quad 0x00AF9A000000FFFF   # 64-bit code: L=1, D=0
    .quad 0x00CF92000000FFFF   # data
gdt64_ptr:
    .word . - gdt64 - 1
    .quad gdt64

.section .text
.code32
.global _start
.type _start, @function
_start:
    cli
    # Identity-map first 64 MiB with 2 MiB pages
    # (kernel VMA == physical 1M, bump heap at physical 16M-48M).
    movl $pdpt, %eax
    orl  $0x3, %eax              # present | writable
    movl %eax, pml4
    movl $0, pml4 + 4
    movl $pd, %eax
    orl  $0x3, %eax
    movl %eax, pdpt
    movl $0, pdpt + 4
    # PD entries 0..31 -> 32 x 2 MiB identity pages
    movl $0, %ecx
.pd_fill:
    movl %ecx, %eax
    shll $21, %eax               # index * 2 MiB
    orl  $0x83, %eax             # PS | RW | P
    movl %eax, pd(,%ecx,8)
    movl $0, pd+4(,%ecx,8)
    incl %ecx
    cmpl $32, %ecx
    jl   .pd_fill

    movl $pml4, %eax
    movl %eax, %cr3

    # CR4: PAE + OSFXSR + OSXMMEXCPT (SSE for kernel doubles)
    movl %cr4, %eax
    orl  $(1 << 5) | (1 << 9) | (1 << 10), %eax
    movl %eax, %cr4

    # EFER.LME + EFER.NXE (hardening maps pages with PTE_NX)
    movl $0xC0000080, %ecx
    rdmsr
    orl  $(1 << 8) | (1 << 11), %eax
    wrmsr

    # CR0: PG + MP, clear EM
    movl %cr0, %eax
    orl  $(1 << 31) | (1 << 1), %eax
    andl $~(1 << 2), %eax
    movl %eax, %cr0

    lgdt gdt64_ptr
    ljmp $0x08, $_start64

.code64
_start64:
    movw $0x10, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %ss
    movw %ax, %fs
    movw %ax, %gs
    mov $stack_top, %rsp
    call kernel_main
    cli
.hang:
    hlt
    jmp .hang
.size _start, .-_start

.section .note.GNU-stack,"",@progbits
