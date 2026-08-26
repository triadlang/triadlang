ENTRY(kmain)

KERNEL_VMA = 0xFFFFFFFF80000000;
KERNEL_LMA = 0x100000;

SECTIONS
{
    . = KERNEL_LMA;
    
    .boot :
    {
        *(.boot)
        *(.boot.*)
    }
    
    . = KERNEL_VMA;
    
    .text ALIGN(4K) : AT(ADDR(.text) - KERNEL_VMA + KERNEL_LMA)
    {
        _text_start = .;
        *(.text)
        *(.text.*)
        _text_end = .;
    }
    
    .rodata ALIGN(4K) : AT(ADDR(.rodata) - KERNEL_VMA + KERNEL_LMA)
    {
        _rodata_start = .;
        *(.rodata)
        *(.rodata.*)
        _rodata_end = .;
    }
    
    .data ALIGN(4K) : AT(ADDR(.data) - KERNEL_VMA + KERNEL_LMA)
    {
        _data_start = .;
        *(.data)
        *(.data.*)
        _data_end = .;
    }
    
    .bss ALIGN(4K) : AT(ADDR(.bss) - KERNEL_VMA + KERNEL_LMA)
    {
        _bss_start = .;
        *(COMMON)
        *(.bss)
        *(.bss.*)
        _bss_end = .;
    }
    
    . = ALIGN(4K);
    _kernel_end = .;
    
    /DISCARD/ :
    {
        *(.comment)
        *(.eh_frame)
        *(.note.*)
    }
}