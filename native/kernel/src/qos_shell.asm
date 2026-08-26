section .text
bits 64

global qos_shell_start
global qos_shell_init
global qos_shell_run_command

extern qos_init
extern qos_create_process
extern qos_evolve_all
extern qos_measure_crystallinity
extern qos_llm_init
extern qos_llm_create
extern qos_llm_generate
extern qos_llm_suggest_command
extern qos_llm_absorb
extern vga_puts
extern vga_putc
extern vga_clear
extern keyboard_get_key
extern keyboard_has_key
extern heap_alloc
extern heap_free

QOS_CMD_MAX equ 512
QOS_HIST_MAX equ 64

qos_shell_init:
    call qos_init
    call qos_llm_init
    
    mov qword [rel qos_process_id], -1
    
    ret

qos_shell_start:
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    mov rdi, .welcome_msg
    call vga_puts
    
    mov rdi, .status_msg
    call vga_puts
    
.shell_loop:
    mov rdi, .prompt_msg
    call vga_puts
    
    xor ecx, ecx
    mov rdi, qos_cmd_buffer
    
.input_loop:
    call keyboard_has_key
    test al, al
    jz .input_loop
    
    call keyboard_get_key
    test al, al
    jz .input_loop
    
    cmp al, 13
    je .execute_cmd
    
    cmp al, 8
    je .backspace
    
    cmp al, 9
    je .autocomplete
    
    cmp ecx, QOS_CMD_MAX - 1
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
    
.autocomplete:
    push rcx
    push rdi
    
    mov rdi, qos_cmd_buffer
    mov rsi, qos_suggestion
    mov rdx, 64
    call qos_llm_suggest_command
    
    pop rdi
    pop rcx
    
    jmp .input_loop
    
.execute_cmd:
    mov byte [rdi], 0
    
    mov rdi, .newline_msg
    call vga_puts
    
    mov rdi, qos_cmd_buffer
    call qos_shell_run_command
    
    call qos_update_system
    
    jmp .shell_loop
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

qos_shell_run_command:
    push rbx
    push r12
    
    mov r12, rdi
    
    mov al, [r12]
    test al, al
    jz .done
    
    mov rdi, .help_cmd
    call qos_match_cmd
    test al, al
    jnz .cmd_help
    
    mov rdi, .ls_cmd
    call qos_match_cmd
    test al, al
    jnz .cmd_ls
    
    mov rdi, .ps_cmd
    call qos_match_cmd
    test al, al
    jnz .cmd_ps
    
    mov rdi, .run_cmd
    call qos_match_cmd
    test al, al
    jnz .cmd_run
    
    mov rdi, .eval_cmd
    call qos_match_cmd
    test al, al
    jnz .cmd_eval
    
    mov rdi, .mem_cmd
    call qos_match_cmd
    test al, al
    jnz .cmd_mem
    
    mov rdi, .status_cmd
    call qos_match_cmd
    test al, al
    jnz .cmd_status
    
    mov rdi, .clear_cmd
    call qos_match_cmd
    test al, al
    jnz .cmd_clear
    
    mov rdi, .exit_cmd
    call qos_match_cmd
    test al, al
    jnz .cmd_exit
    
    mov rdi, .unknown_msg
    call vga_puts
    mov rdi, r12
    call vga_puts
    mov rdi, .newline_msg
    call vga_puts
    
.done:
    pop r12
    pop rbx
    ret
    
.cmd_help:
    mov rdi, .help_text
    call vga_puts
    jmp .done
    
.cmd_ls:
    mov rdi, .ls_text
    call vga_puts
    jmp .done
    
.cmd_ps:
    call qos_print_processes
    jmp .done
    
.cmd_run:
    mov rdi, .run_text
    call vga_puts
    
    add r12, 4
    call qos_run_file
    jmp .done
    
.cmd_eval:
    add r12, 5
    call qos_eval_code
    jmp .done
    
.cmd_mem:
    call qos_print_memory
    jmp .done
    
.cmd_status:
    call qos_print_status
    jmp .done
    
.cmd_clear:
    call vga_clear
    jmp .done
    
.cmd_exit:
    mov rdi, .exit_text
    call vga_puts
    jmp .done

qos_match_cmd:
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

qos_update_system:
    push rax
    
    mov rdi, 0.01
    call qos_evolve_all
    
    mov eax, [rel qos_process_id]
    cmp eax, -1
    je .done
    
    mov rdi, rax
    call qos_measure_crystallinity
    
.done:
    pop rax
    ret

qos_run_file:
    push rbx
    push r12
    
    mov r12, rdi
    
    mov rdi, .running_msg
    call vga_puts
    mov rdi, r12
    call vga_puts
    mov rdi, .newline_msg
    call vga_puts
    
    mov rdi, r12
    mov esi, 256
    mov edx, 10.0
    call qos_create_process
    
    mov [rel qos_process_id], eax
    
    mov rdi, r12
    mov esi, rax
    call qos_llm_absorb
    
    pop r12
    pop rbx
    ret

qos_eval_code:
    push r12
    
    mov r12, rdi
    
    mov rdi, .eval_msg
    call vga_puts
    
    mov rdi, r12
    call qos_llm_absorb
    
    mov rax, [rel qos_process_id]
    cmp eax, -1
    je .done
    
    push r12
    mov rdi, .result_msg
    call vga_puts
    
    mov rdi, .done_msg
    call vga_puts
    pop r12
    
.done:
    pop r12
    ret

qos_print_processes:
    push rbx
    
    mov rdi, .ps_header
    call vga_puts
    
    mov eax, [rel qos_process_id]
    cmp eax, -1
    je .done
    
    mov rdi, .ps_entry
    call vga_puts
    
.done:
    pop rbx
    ret

qos_print_memory:
    push rbx
    
    mov rdi, .mem_header
    call vga_puts
    
    mov rdi, .mem_entry
    call vga_puts
    
    pop rbx
    ret

qos_print_status:
    push rbx
    
    mov rdi, .status_header
    call vga_puts
    
    mov rdi, .status_entry
    call vga_puts
    
    pop rbx
    ret

section .rodata

welcome_msg:
    db "QuantumOS v1.0", 13, 10
    db "================", 13, 10
    db "Sistema operacional com campo quântico nativo", 13, 10, 0

status_msg:
    db "Motor: Triad Solver (P1+P2+P3)", 13, 10
    db "LLM: Integrado para auto-otimização", 13, 10, 0

prompt_msg:
    db "qos> ", 0

newline_msg:
    db 13, 10, 0

help_cmd: db "help", 0
ls_cmd: db "ls", 0
ps_cmd: db "ps", 0
run_cmd: db "run", 0
eval_cmd: db "eval", 0
mem_cmd: db "mem", 0
status_cmd: db "status", 0
clear_cmd: db "clear", 0
exit_cmd: db "exit", 0

help_text:
    db "Comandos:", 13, 10
    db "  help   - Mostra esta ajuda", 13, 10
    db "  ls     - Lista arquivos", 13, 10
    db "  ps     - Lista processos (substratos)", 13, 10
    db "  run    - Executa arquivo .tri", 13, 10
    db "  eval   - Avalia código", 13, 10
    db "  mem    - Status de memória", 13, 10
    db "  status - Status do sistema", 13, 10
    db "  clear  - Limpa tela", 13, 10
    db "  exit   - Sai do shell", 13, 10, 0

ls_text:
    db "kernel.tri    config.sys", 13, 10
    db "init.tri      quantum.tri", 13, 10, 0

ps_header:
    db "PID  NOME            CRYST   ENERGY", 13, 10, 0

ps_entry:
    db "0    kernel          0.95    2.34", 13, 10, 0

mem_header:
    db "MEMÓRIA DO CAMPO:", 13, 10, 0

mem_entry:
    db "  ψ (wavefunction): 256 KB", 13, 10
    db "  y (memory field):  256 KB", 13, 10
    db "  LLM embeddings:    16 KB", 13, 10
    db "  Total:             528 KB", 13, 10, 0

status_header:
    db "STATUS DO SISTEMA:", 13, 10, 0

status_entry:
    db "  Processos ativos: 1", 13, 10
    db "  Estado: CRISTALIZADO", 13, 10
    db "  Equilíbrio: Natural", 13, 10, 0

running_msg:
    db "Executando: ", 0

eval_msg:
    db "Avaliando código...", 13, 10, 0

result_msg:
    db "Resultado: ", 0

done_msg:
    db "[OK]", 13, 10, 0

unknown_msg:
    db "Comando não reconhecido: ", 0

exit_text:
    db "Saindo...", 13, 10, 0

section .bss
align 16

qos_cmd_buffer:
    resb QOS_CMD_MAX

qos_suggestion:
    resb 64

qos_process_id:
    resd 1

qos_llm_handle:
    resq 1