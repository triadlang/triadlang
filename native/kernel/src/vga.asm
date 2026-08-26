section .text
bits 64

global vga_init
global vga_putc
global vga_puts
global vga_clear
global vga_set_color
global vga_get_cursor
global vga_set_cursor
global vga_scroll

VGA_BUFFER equ 0xFFFFFFFF800B8000
VGA_WIDTH equ 80
VGA_HEIGHT equ 25

vga_init:
    mov qword [rel vga_x], 0
    mov qword [rel vga_y], 0
    mov word [rel vga_color], 0x0F00
    call vga_clear
    ret

vga_clear:
    mov rdi, VGA_BUFFER
    mov rax, 0x07200720
    mov rcx, VGA_WIDTH * VGA_HEIGHT / 2
    rep stosq
    mov qword [rel vga_x], 0
    mov qword [rel vga_y], 0
    ret

vga_set_color:
    mov word [rel vga_color], rdi
    ret

vga_get_cursor:
    mov rax, [rel vga_y]
    mov rdx, VGA_WIDTH
    mul rdx
    add rax, [rel vga_x]
    ret

vga_set_cursor:
    mov qword [rel vga_y], rdi
    mov qword [rel vga_x], rsi
    ret

vga_putc:
    cmp dil, 10
    je .newline
    cmp dil, 13
    je .carriage
    cmp dil, 8
    je .backspace
    
    call vga_get_cursor
    mov rdi, VGA_BUFFER
    shl rax, 1
    add rdi, rax
    
    mov al, sil
    mov ah, [rel vga_color]
    mov word [rdi], ax
    
    inc qword [rel vga_x]
    mov rax, [rel vga_x]
    cmp rax, VGA_WIDTH
    jl .done
    
    mov qword [rel vga_x], 0
    inc qword [rel vga_y]
    mov rax, [rel vga_y]
    cmp rax, VGA_HEIGHT
    jl .done
    call vga_scroll
    dec qword [rel vga_y]
    jmp .done
    
.newline:
    mov qword [rel vga_x], 0
    inc qword [rel vga_y]
    mov rax, [rel vga_y]
    cmp rax, VGA_HEIGHT
    jl .done
    call vga_scroll
    dec qword [rel vga_y]
    jmp .done
    
.carriage:
    mov qword [rel vga_x], 0
    jmp .done
    
.backspace:
    mov rax, [rel vga_x]
    test rax, rax
    jz .done
    dec qword [rel vga_x]
    call vga_get_cursor
    mov rdi, VGA_BUFFER
    shl rax, 1
    add rdi, rax
    mov word [rdi], 0x0720
    
.done:
    ret

vga_puts:
    push rbp
    mov rbp, rsp
    push rbx
    
    mov rbx, rdi
    
.loop:
    movzx rax, byte [rbx]
    test al, al
    jz .done
    
    movzx rdi, al
    call vga_putc
    
    inc rbx
    jmp .loop
    
.done:
    pop rbx
    pop rbp
    ret

vga_scroll:
    push rdi
    push rsi
    push rcx
    
    mov rdi, VGA_BUFFER
    mov rsi, VGA_BUFFER + VGA_WIDTH * 2
    mov rcx, (VGA_HEIGHT - 1) * VGA_WIDTH / 4
    rep movsq
    
    mov rdi, VGA_BUFFER + (VGA_HEIGHT - 1) * VGA_WIDTH * 2
    mov rax, 0x07200720
    mov rcx, VGA_WIDTH / 2
    rep stosq
    
    pop rcx
    pop rsi
    pop rdi
    ret

section .data
align 8

vga_x:
    dq 0

vga_y:
    dq 0

vga_color:
    dw 0x0F00