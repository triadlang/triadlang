section .text
bits 64

global isr_divide_error
global isr_debug
global isr_nmi
global isr_breakpoint
global isr_overflow
global isr_bound_range
global isr_invalid_opcode
global isr_device_not_available
global isr_double_fault
global isr_invalid_tss
global isr_segment_not_present
global isr_stack_segment
global isr_general_protection
global isr_page_fault
global isr_floating_point
global isr_alignment_check
global isr_machine_check
global isr_simd_exception

isr_divide_error:
    push 0
    push 0
    jmp isr_common_stub

isr_debug:
    push 0
    push 1
    jmp isr_common_stub

isr_nmi:
    push 0
    push 2
    jmp isr_common_stub

isr_breakpoint:
    push 0
    push 3
    jmp isr_common_stub

isr_overflow:
    push 0
    push 4
    jmp isr_common_stub

isr_bound_range:
    push 0
    push 5
    jmp isr_common_stub

isr_invalid_opcode:
    push 0
    push 6
    jmp isr_common_stub

isr_device_not_available:
    push 0
    push 7
    jmp isr_common_stub

isr_double_fault:
    push 8
    jmp isr_common_stub

isr_invalid_tss:
    push 10
    jmp isr_common_stub

isr_segment_not_present:
    push 11
    jmp isr_common_stub

isr_stack_segment:
    push 12
    jmp isr_common_stub

isr_general_protection:
    push 13
    jmp isr_common_stub

isr_page_fault:
    push 14
    jmp isr_common_stub

isr_floating_point:
    push 0
    push 16
    jmp isr_common_stub

isr_alignment_check:
    push 17
    jmp isr_common_stub

isr_machine_check:
    push 0
    push 18
    jmp isr_common_stub

isr_simd_exception:
    push 0
    push 19
    jmp isr_common_stub

isr_common_stub:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    
    mov rdi, rsp
    call isr_handler
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    pop rbp
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    
    add rsp, 16
    iretq

section .data

isr_handler_table:
    times 256 dq 0