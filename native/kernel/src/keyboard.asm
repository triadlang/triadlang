section .text
bits 64

global keyboard_init
global keyboard_get_key
global keyboard_has_key

keyboard_init:
    mov al, 0xFF
    out 0x21, al
    
    in al, 0x60
    
    mov al, 0xAE
    out 0x64, al
    
    mov al, 0xF4
    out 0x60, al
    
    xor al, al
    out 0x21, al
    
    ret

keyboard_has_key:
    mov rax, [rel kbd_buffer_head]
    mov rcx, [rel kbd_buffer_tail]
    cmp rax, rcx
    setne al
    ret

keyboard_get_key:
    mov rax, [rel kbd_buffer_head]
    mov rcx, [rel kbd_buffer_tail]
    cmp rax, rcx
    je .empty
    
    mov rbx, [rel kbd_buffer_tail]
    movzx rax, byte [rel kbd_buffer + rbx]
    
    inc qword [rel kbd_buffer_tail]
    and qword [rel kbd_buffer_tail], 0xFF
    
    ret
    
.empty:
    xor eax, eax
    ret

global keyboard_handler
keyboard_handler:
    push rax
    push rbx
    
    in al, 0x60
    
    mov rbx, [rel kbd_buffer_head]
    mov [rel kbd_buffer + rbx], al
    inc qword [rel kbd_buffer_head]
    and qword [rel kbd_buffer_head], 0xFF
    
    mov al, 0x20
    out 0x20, al
    
    pop rbx
    pop rax
    iretq

section .bss
align 16

kbd_buffer:
    resb 256

kbd_buffer_head:
    resq 1

kbd_buffer_tail:
    resq 1

kbd_shift:
    resb 1

kbd_ctrl:
    resb 1

kbd_alt:
    resb 1

section .rodata

keymap_lower:
    db 0, 27, '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 8, 9
    db 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', 13, 0, 'a', 's'
    db 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', 39, '`', 0, 92, 'z', 'x', 'c', 'v'
    db 'b', 'n', 'm', ',', '.', '/', 0, '*', 0, ' ', 0, 0, 0, 0, 0, 0
    times 128 db 0

keymap_upper:
    db 0, 27, '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', 8, 9
    db 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', 13, 0, 'A', 'S'
    db 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', 34, '~', 0, '|', 'Z', 'X', 'C', 'V'
    db 'B', 'N', 'M', '<', '>', '?', 0, '*', 0, ' ', 0, 0, 0, 0, 0, 0
    times 128 db 0