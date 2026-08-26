section .text
bits 64

global ata_init
global ata_read_sector
global ata_write_sector
global ata_detect

ATA_PRIMARY_IO equ 0x1F0
ATA_PRIMARY_CTRL equ 0x3F6
ATA_SECONDARY_IO equ 0x170
ATA_SECONDARY_CTRL equ 0x376

ATA_CMD_READ equ 0x20
ATA_CMD_WRITE equ 0x30
ATA_CMD_IDENTIFY equ 0xEC

ata_init:
    call ata_detect
    
    mov dx, ATA_PRIMARY_CTRL
    mov al, 0
    out dx, al
    
    call ata_delay
    
    ret

ata_detect:
    mov dx, ATA_PRIMARY_IO + 6
    mov al, 0xA0
    out dx, al
    
    call ata_delay
    
    mov dx, ATA_PRIMARY_IO + 2
    mov al, 0
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 3
    mov al, 0
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 4
    mov al, 0
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 5
    mov al, 0
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 7
    mov al, ATA_CMD_IDENTIFY
    out dx, al
    
    call ataDelay400ns
    
    mov dx, ATA_PRIMARY_IO + 7
    in al, dx
    test al, al
    jz .no_drive
    
    mov dx, ATA_PRIMARY_IO + 4
    in al, dx
    mov ah, al
    mov dx, ATA_PRIMARY_IO + 5
    in al, dx
    cmp ax, 0xEB14
    je .found
    
.no_drive:
    mov qword [rel ata_sectors], 0
    ret
    
.found:
    call ata_read_identify
    ret

ata_read_identify:
    push rbx
    push r12
    
    mov dx, ATA_PRIMARY_IO + 7
.wait:
    in al, dx
    test al, 0x80
    jnz .wait
    
    mov dx, ATA_PRIMARY_IO + 7
    in al, dx
    test al, 1
    jnz .error
    
    mov dx, ATA_PRIMARY_IO + 7
.wait_data:
    in al, dx
    mov ah, al
    and ah, 0x08
    jz .wait_data
    
    mov r12, ata_identify_buf
    
    mov ecx, 256
.read_loop:
    mov dx, ATA_PRIMARY_IO
    in ax, dx
    mov [r12], ax
    add r12, 2
    dec ecx
    jnz .read_loop
    
    mov rax, ata_identify_buf
    mov eax, [rax + 120]
    mov [rel ata_sectors], eax
    
    pop r12
    pop rbx
    ret
    
.error:
    pop r12
    pop rbx
    xor eax, eax
    ret

ata_read_sector:
    push rbx
    push r12
    push r13
    
    mov r12, rdi
    mov r13, rsi
    
    call ata_wait_ready
    
    mov dx, ATA_PRIMARY_IO + 6
    mov al, 0xE0
    or al, r12b
    and al, 0x0F
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 2
    mov al, 1
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 3
    mov al, r12b
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 4
    mov rax, r12
    shr rax, 8
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 5
    mov rax, r12
    shr rax, 16
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 7
    mov al, ATA_CMD_READ
    out dx, al
    
    call ata_wait_data
    
    mov rdi, r13
    mov ecx, 256
.read_loop:
    mov dx, ATA_PRIMARY_IO
    in ax, dx
    mov [rdi], ax
    add rdi, 2
    dec ecx
    jnz .read_loop
    
    pop r13
    pop r12
    pop rbx
    ret

ata_write_sector:
    push rbx
    push r12
    push r13
    
    mov r12, rdi
    mov r13, rsi
    
    call ata_wait_ready
    
    mov dx, ATA_PRIMARY_IO + 6
    mov al, 0xE0
    or al, r12b
    and al, 0x0F
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 2
    mov al, 1
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 3
    mov al, r12b
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 4
    mov rax, r12
    shr rax, 8
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 5
    mov rax, r12
    shr rax, 16
    out dx, al
    
    mov dx, ATA_PRIMARY_IO + 7
    mov al, ATA_CMD_WRITE
    out dx, al
    
    call ata_wait_ready
    
    mov rsi, r13
    mov ecx, 256
.write_loop:
    mov dx, ATA_PRIMARY_IO
    mov ax, [rsi]
    out dx, ax
    add rsi, 2
    dec ecx
    jnz .write_loop
    
    call ata_flush_cache
    
    pop r13
    pop r12
    pop rbx
    ret

ata_wait_ready:
    mov dx, ATA_PRIMARY_CTRL
.wait:
    in al, dx
    test al, 0x80
    jnz .wait
    ret

ata_wait_data:
    mov dx, ATA_PRIMARY_CTRL
.wait:
    in al, dx
    mov ah, al
    and ah, 0x08
    jz .wait
    ret

ataDelay400ns:
    mov dx, ATA_PRIMARY_CTRL
    in al, dx
    in al, dx
    in al, dx
    in al, dx
    ret

ata_flush_cache:
    mov dx, ATA_PRIMARY_IO + 7
    mov al, 0xE7
    out dx, al
    call ata_wait_ready
    ret

ata_delay:
    push rcx
    mov ecx, 10000
.delay_loop:
    nop
    dec ecx
    jnz .delay_loop
    pop rcx
    ret

section .bss
align 16

ata_identify_buf:
    resb 512

ata_sectors:
    resq 1