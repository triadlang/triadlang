.section .text
.global triad_ctx_save
.global triad_ctx_restore
.global triad_ctx_switch
.global triad_ctx_init_thread

triad_ctx_save:
    movq %r15, 0(%rdi)
    movq %r14, 8(%rdi)
    movq %r13, 16(%rdi)
    movq %r12, 24(%rdi)
    movq %r11, 32(%rdi)
    movq %r10, 40(%rdi)
    movq %r9,  48(%rdi)
    movq %r8,  56(%rdi)
    movq %rdi, 64(%rdi)
    movq %rsi, 72(%rdi)
    movq %rbp, 80(%rdi)
    movq %rdx, 88(%rdi)
    movq %rcx, 96(%rdi)
    movq %rbx, 104(%rdi)
    movq %rax, 112(%rdi)
    movq (%rsp), %rax
    movq %rax, 120(%rdi)
    leaq 8(%rsp), %rax
    movq %rax, 128(%rdi)
    pushfq
    popq %rax
    movq %rax, 136(%rdi)
    ret

triad_ctx_restore:
    movq 0(%rdi), %r15
    movq 8(%rdi), %r14
    movq 16(%rdi), %r13
    movq 24(%rdi), %r12
    movq 32(%rdi), %r11
    movq 40(%rdi), %r10
    movq 48(%rdi), %r9
    movq 56(%rdi), %r8
    movq 72(%rdi), %rsi
    movq 80(%rdi), %rbp
    movq 88(%rdi), %rdx
    movq 96(%rdi), %rcx
    movq 104(%rdi), %rbx
    movq 112(%rdi), %rax
    movq 136(%rdi), %r8
    pushq %r8
    popfq
    movq 120(%rdi), %r8
    movq 128(%rdi), %rsp
    pushq %r8
    movq 64(%rdi), %rdi
    ret

triad_ctx_switch:
    pushq %rbp
    movq %rsp, %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    movq %rsp, (%rdi)
    movq (%rsi), %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret

triad_ctx_init_thread:
    movq %rsi, 120(%rdi)
    leaq 8(%rsi), %rax
    movq %rax, 128(%rdi)
    movq $0x202, %rax
    movq %rax, 136(%rdi)
    movq $0, 64(%rdi)
    ret

.section .note.GNU-stack,"",@progbits
