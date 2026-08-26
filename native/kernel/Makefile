UNAME_S := $(shell uname -s)
CC = gcc
AS = as
LD = ld

ifeq ($(UNAME_S),Darwin)
  KERNEL_CC_X86_ELF := $(shell command -v x86_64-none-elf-gcc 2>/dev/null)
  ifeq ($(KERNEL_CC_X86_ELF),)
    KERNEL_CC_X86_ELF := $(shell command -v x86_64-elf-gcc 2>/dev/null)
  endif
  KERNEL_AS_X86_ELF := $(shell command -v x86_64-none-elf-as 2>/dev/null)
  ifeq ($(KERNEL_AS_X86_ELF),)
    KERNEL_AS_X86_ELF := $(shell command -v x86_64-elf-as 2>/dev/null)
  endif
  KERNEL_LD_X86_ELF := $(shell command -v x86_64-none-elf-ld 2>/dev/null)
  ifeq ($(KERNEL_LD_X86_ELF),)
    KERNEL_LD_X86_ELF := $(shell command -v x86_64-elf-ld 2>/dev/null)
  endif
  KERNEL_OBJCOPY_X86_ELF := $(shell command -v x86_64-none-elf-objcopy 2>/dev/null)
  ifeq ($(KERNEL_OBJCOPY_X86_ELF),)
    KERNEL_OBJCOPY_X86_ELF := $(shell command -v x86_64-elf-objcopy 2>/dev/null)
  endif

  ifneq ($(KERNEL_CC_X86_ELF),)
    CC := $(KERNEL_CC_X86_ELF)
  endif
  ifneq ($(KERNEL_AS_X86_ELF),)
    AS := $(KERNEL_AS_X86_ELF)
  endif
  ifneq ($(KERNEL_LD_X86_ELF),)
    LD := $(KERNEL_LD_X86_ELF)
  endif
  ifneq ($(KERNEL_OBJCOPY_X86_ELF),)
    OBJCOPY := $(KERNEL_OBJCOPY_X86_ELF)
  else
    OBJCOPY := $(shell command -v objcopy 2>/dev/null)
  endif

  ifeq ($(CC),gcc)
    ifeq ($(findstring -elf, $(CC)),)
    $(warning Found no x86_64-elf cross toolchain on macOS)
    $(error To build native/kernel on macOS, install an x86_64-elf cross toolchain (gcc/as/ld/objcopy) and retry. Example: brew install x86_64-elf-gcc x86_64-elf-binutils, then run this target again.)
    endif
  endif
endif

CFLAGS = -ffreestanding -nostdlib -fno-stack-protector \
         -fno-pic \
         -Wall -Wextra -O2 \
         -I include -D__KERNEL__

ifeq ($(UNAME_S),Linux)
  CFLAGS += -mno-red-zone -mno-mmx -mno-avx
endif

ASFLAGS = --64 -I include/
ifeq ($(UNAME_S),Darwin)
  ASFLAGS = --64 -I include/
endif

LDFLAGS = -nostdlib -static -T linker.ld -z max-page-size=0x1000

KERNEL_C = triad_kernel_main.c triad_gdt_idt.c triad_isr.c triad_pic.c \
          triad_pit.c triad_serial.c triad_keyboard.c triad_mm.c \
          triad_string.c \
          triad_paging.c triad_vfs.c triad_sched.c triad_syscall.c \
          triad_ai.c triad_equilibrium.c triad_sandbox.c \
          triad_interpreter.c triad_tri_loader.c triad_shell.c \
          triad_user.c triad_ata.c triad_disk.c triad_net.c \
          triad_rtl8139.c triad_runtime_svc.c \
          triad_stack_protect.c triad_aslr.c triad_crypto.c \
          triad_audit.c triad_firewall.c triad_hardening.c \
          triad_signal.c

KERNEL_ASM = boot.asm triad_isr.asm triad_context_switch.asm

OBJDIR = obj

all: kernel.bin

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: %.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/asm_%.o: %.asm | $(OBJDIR)
	$(AS) $(ASFLAGS) $< -o $@

kernel.bin: $(patsubst %.c,$(OBJDIR)/%.o,$(KERNEL_C)) $(patsubst %.asm,$(OBJDIR)/asm_%.o,$(KERNEL_ASM)) linker.ld
	@echo "Linking..."
	$(LD) $(LDFLAGS) -o $@ $(OBJDIR)/*.o
	$(OBJCOPY) -O elf32-i386 $@ kernel32.bin
	@echo "Done: kernel.bin (elf64) + kernel32.bin (multiboot elf32)"

run: kernel.bin
	qemu-system-x86_64 -kernel kernel32.bin -m 256M -cpu qemu64 -nographic

debug: kernel.bin
	qemu-system-x86_64 -kernel kernel.bin -m 256M -cpu qemu64 -s -S &
	gdb -ex "target remote localhost:1234" -ex "break kernel_main" -ex "continue"

iso: kernel.bin
	@mkdir -p isodir/boot/grub
	cp kernel.bin isodir/boot/kernel.bin
	echo 'set timeout=0' > isodir/boot/grub/grub.cfg
	echo 'set default=0' >> isodir/boot/grub/grub.cfg
	echo 'menuentry "TriadOS" {' >> isodir/boot/grub/grub.cfg
	echo '  multiboot /boot/kernel.bin' >> isodir/boot/grub/grub.cfg
	echo '}' >> isodir/boot/grub/grub.cfg
	grub-mkrescue -o triados.iso isodir

clean:
	rm -rf $(OBJDIR) kernel.bin triados.iso isodir

.PHONY: all run debug iso clean
