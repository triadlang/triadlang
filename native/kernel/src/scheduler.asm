section .text
bits 64

global scheduler_init
global scheduler_add_process
global scheduler_remove_process
global scheduler_yield
global scheduler_tick
global scheduler_get_current

MAX_PROCESSES equ 64
QUANTUM equ 10

scheduler_init:
    xor eax, eax
    mov [rel process_count], eax
    mov [rel current_process], eax
    
    mov rdi, process_table
    mov ecx, MAX_PROCESSES * 256
    xor eax, eax
.clear_loop:
    mov qword [rdi], eax
    add rdi, 8
    dec ecx
    jnz .clear_loop
    
    mov qword [rel scheduler_lock], 0
    
    ret

scheduler_add_process:
    push rbx
    push r12
    
    mov r12, rdi
    mov rbx, rsi
    
    mov eax, [rel process_count]
    cmp eax, MAX_PROCESSES
    jge .fail
    
    mov ecx, eax
    shl ecx, 8
    lea rdi, [rel process_table + rcx]
    
    mov qword [rdi], 1
    mov qword [rdi + 8], rbx
    mov qword [rdi + 16], r12
    mov qword [rdi + 24], 0
    mov qword [rdi + 40], 0
    mov byte [rdi + 48], 0
    mov byte [rdi + 49], 1
    mov qword [rdi + 56], 0
    
    mov eax, [rel process_count]
    mov dword [rdi + 64], eax
    
    inc dword [rel process_count]
    
    mov eax, [rel process_count]
    dec eax
    
    pop r12
    pop rbx
    ret
    
.fail:
    mov eax, -1
    pop r12
    pop rbx
    ret

scheduler_remove_process:
    push rbx
    
    mov eax, [rel current_process]
    cmp edi, eax
    jne .find_slot
    
    mov ecx, edi
    shl ecx, 8
    lea rbx, [rel process_table + rcx]
    
    mov byte [rbx + 49], 4
    call scheduler_yield
    
    jmp .done
    
.find_slot:
    cmp edi, MAX_PROCESSES
    jge .done
    
    mov ecx, edi
    shl ecx, 8
    lea rbx, [rel process_table + rcx]
    mov byte [rbx + 49], 4
    
.done:
    pop rbx
    ret

scheduler_yield:
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    mov eax, [rel current_process]
    mov ecx, eax
    shl ecx, 8
    lea rbx, [rel process_table + rcx]
    
    mov [rbx + 24], rsp
    mov [rbx + 32], rbp
    mov [rbx + 56], rax
    
.find_next:
    mov eax, [rel current_process]
    inc eax
    cmp eax, [rel process_count]
    jl .check_proc
    xor eax, eax
    
.check_proc:
    mov ecx, eax
    shl ecx, 8
    lea rbx, [rel process_table + rcx]
    
    movzx edx, byte [rbx + 49]
    cmp edx, 1
    je .found
    
    inc eax
    cmp eax, [rel current_process]
    jne .check_proc
    
    mov eax, [rel current_process]
    jmp .found
    
.found:
    mov [rel current_process], eax
    
    mov ecx, eax
    shl ecx, 8
    lea rbx, [rel process_table + rcx]
    
    mov dword [rel time_slice], QUANTUM
    
    mov rsp, [rbx + 24]
    mov rbp, [rbx + 32]
    mov rax, [rbx + 56]
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

scheduler_tick:
    push rax
    
    dec dword [rel time_slice]
    jg .done
    
    mov dword [rel time_slice], QUANTUM
    call scheduler_yield
    
.done:
    pop rax
    ret

scheduler_get_current:
    mov eax, [rel current_process]
    ret

section .bss
align 16

process_table:
    resb MAX_PROCESSES * 256

process_count:
    resd 1

current_process:
    resd 1

time_slice:
    resd 1

scheduler_lock:
    resq 1