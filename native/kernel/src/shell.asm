section .text
bits 64

global shell_start
global shell_execute
global shell_parse_line
global shell_cmd_help
global shell_cmd_clear
global shell_cmd_ls
global shell_cmd_cat
global shell_cmd_run
global shell_cmd_eval
global shell_cmd_mem

extern vga_puts
extern vga_putc
extern vga_clear
extern keyboard_get_key
extern keyboard_has_key
extern triad_kernel_init
extern triad_kernel_parse
extern triad_kernel_eval
extern triad_kernel_run_file
extern heap_alloc
extern heap_free
extern memory_get_total
extern memory_get_used

SHELL_CMD_MAX equ 256
SHELL_HIST_MAX equ 64

shell_start:
    mov rdi, .welcome_msg
    call vga_puts
    
    call shell_init_history
    
.shell_loop:
    mov rdi, .prompt_msg
    call vga_puts
    
    xor ecx, ecx
    mov rdi, shell_input_buf
    
.input_loop:
    call keyboard_has_key
    test al, al
    jz .input_loop
    
    call keyboard_get_key
    test al, al
    jz .input_loop
    
    cmp al, 13
    je .execute_line
    
    cmp al, 8
    je .backspace
    
    cmp al, 27
    je .escape_seq
    
    cmp ecx, SHELL_CMD_MAX - 1
    jge .input_loop
    
    mov [rdi], al
    inc rdi
    inc ecx
    
    push rcx
    movzx rdi, al
    call vga_putc
    pop rcx
    
    jmp .input_loop
    
.backspace:
    test ecx, ecx
    jz .input_loop
    
    dec rdi
    dec ecx
    
    push rcx
    mov rdi, 8
    call vga_putc
    pop rcx
    
    jmp .input_loop
    
.escape_seq:
    call keyboard_get_key
    cmp al, '['
    jne .input_loop
    
    call keyboard_get_key
    cmp al, 'A'
    je .history_up
    cmp al, 'B'
    je .history_down
    
    jmp .input_loop
    
.history_up:
    mov rax, [rel shell_hist_idx]
    test rax, rax
    jz .input_loop
    
    dec rax
    mov [rel shell_hist_idx], rax
    
    call shell_redraw
    
    jmp .input_loop
    
.history_down:
    mov rax, [rel shell_hist_idx]
    mov rbx, [rel shell_hist_count]
    cmp rax, rbx
    jge .input_loop
    
    inc rax
    mov [rel shell_hist_idx], rax
    
    call shell_redraw
    
    jmp .input_loop
    
.execute_line:
    mov byte [rdi], 0
    
    mov rdi, shell_input_buf
    call shell_execute
    
    call shell_save_history
    
    jmp .shell_loop
    
    ret

shell_execute:
    push rbx
    push r12
    push r13
    
    mov r12, rdi
    
    mov al, [r12]
    test al, al
    jz .done
    
    mov rdi, .help_cmd
    call shell_match_cmd
    test al, al
    jnz .cmd_help
    
    mov rdi, .clear_cmd
    call shell_match_cmd
    test al, al
    jnz .cmd_clear
    
    mov rdi, .ls_cmd
    call shell_match_cmd
    test al, al
    jnz .cmd_ls
    
    mov rdi, .cat_cmd
    call shell_match_cmd
    test al, al
    jnz .cmd_cat
    
    mov rdi, .run_cmd
    call shell_match_cmd
    test al, al
    jnz .cmd_run
    
    mov rdi, .eval_cmd
    call shell_match_cmd
    test al, al
    jnz .cmd_eval
    
    mov rdi, .mem_cmd
    call shell_match_cmd
    test al, al
    jnz .cmd_mem
    
    mov rdi, .unknown_msg
    call vga_puts
    mov rdi, r12
    call vga_puts
    mov rdi, .newline_msg
    call vga_puts
    
.done:
    pop r13
    pop r12
    pop rbx
    ret
    
.cmd_help:
    call shell_cmd_help
    jmp .done
    
.cmd_clear:
    call shell_cmd_clear
    jmp .done
    
.cmd_ls:
    call shell_cmd_ls
    jmp .done
    
.cmd_cat:
    call shell_cmd_cat
    jmp .done
    
.cmd_run:
    call shell_cmd_run
    jmp .done
    
.cmd_eval:
    call shell_cmd_eval
    jmp .done
    
.cmd_mem:
    call shell_cmd_mem
    jmp .done

shell_match_cmd:
    mov rsi, r12
    
.match_loop:
    mov al, [rdi]
    test al, al
    jz .check_space
    
    mov bl, [rsi]
    cmp al, bl
    jne .no_match
    
    inc rdi
    inc rsi
    jmp .match_loop
    
.check_space:
    mov al, [rsi]
    cmp al, ' '
    je .match
    cmp al, 0
    je .match
    cmp al, 13
    je .match
    
.no_match:
    xor eax, eax
    ret
    
.match:
    mov al, 1
    ret

shell_cmd_help:
    push rbx
    
    mov rdi, .help_msg1
    call vga_puts
    mov rdi, .help_msg2
    call vga_puts
    mov rdi, .help_msg3
    call vga_puts
    mov rdi, .help_msg4
    call vga_puts
    mov rdi, .help_msg5
    call vga_puts
    mov rdi, .help_msg6
    call vga_puts
    mov rdi, .help_msg7
    call vga_puts
    
    pop rbx
    ret

shell_cmd_clear:
    call vga_clear
    ret

shell_cmd_ls:
    mov rdi, .ls_msg
    call vga_puts
    ret

shell_cmd_cat:
    mov rdi, .cat_msg
    call vga_puts
    ret

shell_cmd_run:
    mov rdi, .run_msg
    call vga_puts
    ret

shell_cmd_eval:
    push rbx
    push r12
    
    mov r12, shell_input_buf
    add r12, 5
    
    mov rdi, r12
    mov rsi, triad_eval_output
    mov rdx, 4096
    call triad_kernel_eval
    
    mov rdi, triad_eval_output
    call vga_puts
    mov rdi, .newline_msg
    call vga_puts
    
    pop r12
    pop rbx
    ret

shell_cmd_mem:
    call memory_get_total
    mov rbx, rax
    
    call memory_get_used
    mov rcx, rax
    
    push rbx
    push rcx
    
    mov rdi, .mem_msg1
    call vga_puts
    
    pop rcx
    pop rbx
    
    push rbx
    mov rdi, rbx
    call shell_print_num
    
    mov rdi, .mem_msg2
    call vga_puts
    
    mov rdi, .mem_msg3
    call vga_puts
    
    ret

shell_print_num:
    push rbx
    push rcx
    push rdx
    
    mov rbx, 10
    xor ecx, ecx
    
.divide_loop:
    xor edx, edx
    div rbx
    push rdx
    inc ecx
    test rax, rax
    jnz .divide_loop
    
.print_loop:
    pop rax
    add al, '0'
    movzx rdi, al
    call vga_putc
    dec ecx
    jnz .print_loop
    
    pop rdx
    pop rcx
    pop rbx
    ret

shell_save_history:
    push rbx
    push r12
    
    mov rax, [rel shell_hist_count]
    cmp rax, SHELL_HIST_MAX
    jge .done
    
    mov rbx, shell_hist_buf
    shl rax, 8
    add rbx, rax
    
    mov r12, shell_input_buf
    
.copy_loop:
    mov al, [r12]
    mov [rbx], al
    test al, al
    jz .copied
    inc r12
    inc rbx
    jmp .copy_loop
    
.copied:
    inc qword [rel shell_hist_count]
    mov rax, [rel shell_hist_count]
    mov [rel shell_hist_idx], rax
    
.done:
    pop r12
    pop rbx
    ret

shell_init_history:
    mov qword [rel shell_hist_count], 0
    mov qword [rel shell_hist_idx], 0
    ret

shell_redraw:
    ret

section .rodata

welcome_msg:
    db "TriadOS v0.3 - Native TriadLang Runtime", 13, 10
    db "=========================================", 13, 10
    db "Type 'help' for commands.", 13, 10, 0

prompt_msg:
    db "triad> ", 0

newline_msg:
    db 13, 10, 0

help_cmd:
    db "help", 0
clear_cmd:
    db "clear", 0
ls_cmd:
    db "ls", 0
cat_cmd:
    db "cat", 0
run_cmd:
    db "run", 0
eval_cmd:
    db "eval", 0
mem_cmd:
    db "mem", 0

unknown_msg:
    db "Unknown command: ", 0

ls_msg:
    db "Files: kernel.bin config.sys", 13, 10, 0

cat_msg:
    db "Usage: cat <filename>", 13, 10, 0

run_msg:
    db "Running program...", 13, 10, 0

mem_msg1:
    db "Memory: ", 0
mem_msg2:
    db " KB used", 13, 10, 0
mem_msg3:
    db "Memory info displayed.", 13, 10, 0

help_msg1:
    db "Available commands:", 13, 10, 0
help_msg2:
    db "  help  - Show this help", 13, 10, 0
help_msg3:
    db "  clear - Clear screen", 13, 10, 0
help_msg4:
    db "  ls    - List files", 13, 10, 0
help_msg5:
    db "  cat   - View file", 13, 10, 0
help_msg6:
    db "  eval  - Evaluate TriadLang", 13, 10, 0
help_msg7:
    db "  mem   - Memory status", 13, 10, 0

section .bss
align 16

shell_input_buf:
    resb SHELL_CMD_MAX

shell_hist_buf:
    resb SHELL_HIST_MAX * SHELL_CMD_MAX

shell_hist_count:
    resq 1

shell_hist_idx:
    resq 1

triad_eval_output:
    resb 4096

triad_file_buffer:
    resb 65536