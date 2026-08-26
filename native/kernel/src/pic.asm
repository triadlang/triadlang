section .text
bits 64

global pic_init
global pic_send_eoi
global pic_set_mask
global pic_clear_mask

pic_init:
    mov al, 0x11
    out 0x20, al
    out 0xA0, al
    
    mov al, 0x20
    out 0x21, al
    mov al, 0x28
    out 0xA1, al
    
    mov al, 0x04
    out 0x21, al
    mov al, 0x02
    out 0xA1, al
    
    mov al, 0x01
    out 0x21, al
    out 0xA1, al
    
    mov al, 0xFF
    out 0x21, al
    mov al, 0xFF
    out 0xA1, al
    
    ret

pic_send_eoi:
    mov al, 0x20
    out 0x20, al
    out 0xA0, al
    ret

pic_set_mask:
    cmp rdi, 8
    jl .master
    sub rdi, 8
    in al, 0xA1
    or al, rdi
    jmp .done
.master:
    in al, 0x21
    or al, rdi
.done:
    ret

pic_clear_mask:
    cmp rdi, 8
    jl .master
    sub rdi, 8
    in al, 0xA1
    mov cl, rdi
    not cl
    and al, cl
    out 0xA1, al
    ret
.master:
    in al, 0x21
    mov cl, rdi
    not cl
    and al, cl
    out 0x21, al
    ret