section .text
bits 64

global vfs_init
global vfs_mount
global vfs_open
global vfs_close
global vfs_read
global vfs_write
global vfs_seek
global vfs_readdir
global vfs_mkdir
global vfs_rmdir
global vfs_unlink
global vfs_stat

VFS_MAX_MOUNTS equ 16
VFS_MAX_FILES equ 128
VFS_MAX_PATH equ 256

vfs_init:
    mov rdi, [rel vfs_mounts]
    mov ecx, VFS_MAX_MOUNTS * 16
    xor eax, eax
    rep stosb
    
    mov rdi, [rel vfs_files]
    mov ecx, VFS_MAX_FILES * 32
    xor eax, eax
    rep stosb
    
    mov qword [rel vfs_next_fd], 3
    mov qword [rel vfs_mount_count], 0
    
    ret

vfs_mount:
    mov rax, [rel vfs_mount_count]
    cmp rax, VFS_MAX_MOUNTS
    jge .fail
    
    mov rbx, vfs_mounts
    shl rax, 4
    add rbx, rax
    
    lea rdi, [rbx]
    mov rsi, rdi
    mov ecx, 64
.copy_path:
    mov al, [rsi]
    mov [rdi], al
    inc rsi
    inc rdi
    test al, al
    jnz .copy_path
    
    mov [rbx + 8], rsi
    
    inc qword [rel vfs_mount_count]
    mov rax, [rel vfs_mount_count]
    ret
    
.fail:
    xor eax, eax
    ret

vfs_open:
    mov rax, [rel vfs_next_fd]
    cmp rax, VFS_MAX_FILES
    jge .fail
    
    mov rbx, vfs_files
    shl rax, 5
    add rbx, rax
    
    mov qword [rbx], rdi
    mov qword [rbx + 8], rsi
    mov qword [rbx + 16], 0
    
    mov rax, [rel vfs_next_fd]
    inc qword [rel vfs_next_fd]
    
    ret
    
.fail:
    mov rax, -1
    ret

vfs_close:
    cmp rdi, VFS_MAX_FILES
    jge .fail
    
    mov rbx, vfs_files
    shl rdi, 5
    add rbx, rdi
    
    mov qword [rbx], 0
    mov qword [rbx + 8], 0
    mov qword [rbx + 16], 0
    
    xor eax, eax
    ret
    
.fail:
    mov rax, -1
    ret

vfs_read:
    cmp rdi, VFS_MAX_FILES
    jge .fail
    
    mov rbx, vfs_files
    shl rdi, 5
    add rbx, rdi
    
    mov rdi, [rbx + 8]
    call disk_read
    
    ret
    
.fail:
    xor eax, eax
    ret

vfs_write:
    cmp rdi, VFS_MAX_FILES
    jge .fail
    
    mov rbx, vfs_files
    shl rdi, 5
    add rbx, rdi
    
    mov rdi, [rbx + 8]
    call disk_write
    
    ret
    
.fail:
    xor eax, eax
    ret

vfs_seek:
    cmp rdi, VFS_MAX_FILES
    jge .fail
    
    mov rbx, vfs_files
    shl rdi, 5
    add rbx, rdi
    
    cmp rsi, 0
    jl .fail
    
    cmp rsi, 4096
    jge .fail
    
    mov [rbx + 16], rsi
    
    xor eax, eax
    ret
    
.fail:
    mov rax, -1
    ret

vfs_readdir:
    xor eax, eax
    ret

vfs_mkdir:
    xor eax, eax
    ret

vfs_rmdir:
    xor eax, eax
    ret

vfs_unlink:
    xor eax, eax
    ret

vfs_stat:
    xor eax, eax
    ret

section .bss
align 16

vfs_mounts:
    resb VFS_MAX_MOUNTS * 16

vfs_files:
    resb VFS_MAX_FILES * 32

vfs_next_fd:
    resq 1

vfs_mount_count:
    resq 1