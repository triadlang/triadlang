section .text
bits 64

global syscall_handler
global syscall_table

extern sys_read
extern sys_write
extern sys_open
extern sys_close
extern sys_exec
extern sys_exit
extern sys_yield
extern sys_getpid
extern sys_mmap
extern sys_munmap
extern sys_time
extern sys_heap_alloc
extern sys_heap_free

syscall_handler:
    swapgs
    
    mov [rel syscall_saved_rsp], rsp
    mov rsp, [rel kernel_stack]
    
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    mov rbp, rsp
    
    cmp rax, 0
    jl .invalid_syscall
    cmp rax, syscall_count
    jge .invalid_syscall
    
    mov rbx, [rel syscall_table + rax * 8]
    call rbx
    
    mov [rel syscall_ret], rax
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    
    mov rsp, [rel syscall_saved_rsp]
    
    swapgs
    iretq
    
.invalid_syscall:
    mov rax, -1
    mov [rel syscall_ret], rax
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    
    mov rsp, [rel syscall_saved_rsp]
    
    swapgs
    iretq

syscall_read:
    mov rdi, rsi
    mov rsi, rdx
    mov rdx, rcx
    call sys_read
    ret

syscall_write:
    mov rdi, rsi
    mov rsi, rdx
    mov rdx, rcx
    call sys_write
    ret

syscall_open:
    mov rdi, rsi
    mov rsi, rdx
    call sys_open
    ret

syscall_close:
    mov rdi, rsi
    call sys_close
    ret

syscall_exec:
    mov rdi, rsi
    mov rsi, rdx
    call sys_exec
    ret

syscall_exit:
    mov rdi, rsi
    call sys_exit
    ret

syscall_yield:
    call sys_yield
    ret

syscall_getpid:
    call sys_getpid
    ret

syscall_mmap:
    mov rdi, rsi
    mov rsi, rdx
    mov rdx, rcx
    call sys_mmap
    ret

syscall_munmap:
    mov rdi, rsi
    mov rsi, rdx
    call sys_munmap
    ret

syscall_time:
    call sys_time
    ret

syscall_heap_alloc:
    mov rdi, rsi
    call sys_heap_alloc
    ret

syscall_heap_free:
    mov rdi, rsi
    mov rsi, rdx
    call sys_heap_free
    ret

section .data
align 16

syscall_table:
    dq syscall_read
    dq syscall_write
    dq syscall_open
    dq syscall_close
    dq syscall_exec
    dq syscall_exit
    dq syscall_yield
    dq syscall_getpid
    dq syscall_mmap
    dq syscall_munmap
    dq syscall_time
    dq syscall_heap_alloc
    dq syscall_heap_free
syscall_count equ ($ - syscall_table) / 8

section .bss
align 16

syscall_saved_rsp:
    resq 1

kernel_stack:
    resq 1

syscall_ret:
    resq 1