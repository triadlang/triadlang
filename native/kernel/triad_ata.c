#include "triad_ata.h"
#include "triad_serial.h"
#include <string.h>

static AtaDrive drives[4];
static uint16_t ata_buses[2] = { ATA_PRIMARY_IO, ATA_SECONDARY_IO };
static uint16_t ata_ctrls[2] = { ATA_PRIMARY_CTRL, ATA_SECONDARY_CTRL };

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ volatile ("outb %0, %1" : : "a"(val), "Nd"(port));
}

static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    __asm__ volatile ("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

static inline void outw(uint16_t port, uint16_t val) {
    __asm__ volatile ("outw %0, %1" : : "a"(val), "Nd"(port));
}

static inline uint16_t inw(uint16_t port) {
    uint16_t ret;
    __asm__ volatile ("inw %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

static int ata_wait_bsy(uint16_t io) {
    for (uint32_t i = 0; i < 100000; i++) {
        if (!(inb(io + ATA_STATUS) & ATA_STATUS_BSY)) return 0;
    }
    return -1;
}

static void ata_soft_reset(uint16_t ctrl) {
    outb(ctrl, 0x04);
    for (int i = 0; i < 1000; i++) {
        do { uint8_t d_; __asm__ volatile ("inb $0x80, %0" : "=a"(d_) :: "memory"); } while (0);
    }
    outb(ctrl, 0x00);
}

static int ata_select_drive(uint16_t io, uint8_t drive) {
    outb(io + ATA_DRIVE, drive == ATA_MASTER ? 0xA0 : 0xB0);
    for (int i = 0; i < 1000; i++) {
        do { uint8_t d_; __asm__ volatile ("inb $0x80, %0" : "=a"(d_) :: "memory"); } while (0);
    }
    return 0;
}

static int ata_poll(uint16_t io) {
    int timeout = 100000;
    while (timeout--) {
        uint8_t status = inb(io + ATA_STATUS);
        if (status & ATA_STATUS_ERR) return -1;
        if (!(status & ATA_STATUS_BSY) && (status & ATA_STATUS_DRQ)) return 0;
    }
    return -1;
}

void triad_ata_init(void) {
    for (int i = 0; i < 4; i++) {
        drives[i].present = false;
    }

    for (int bus = 0; bus < 2; bus++) {
        uint16_t io = ata_buses[bus];
        uint16_t ctrl = ata_ctrls[bus];

        ata_soft_reset(ctrl);

        for (int drive = 0; drive < 2; drive++) {
            int idx = bus * 2 + drive;
            AtaDrive *d = &drives[idx];
            d->io_base = io;
            d->ctrl_base = ctrl;
            d->master = (drive == 0);

            ata_select_drive(io, drive);

            outb(io + ATA_LBA_LOW, 0);
            outb(io + ATA_LBA_MID, 0);
            outb(io + ATA_LBA_HIGH, 0);
            outb(io + ATA_COMMAND, ATA_CMD_IDENTIFY);

            uint8_t status = inb(io + ATA_STATUS);
            if (status == 0 || status == 0xFF) {
                d->present = false;
                continue;
            }

            ata_wait_bsy(io);

            uint8_t lba_mid = inb(io + ATA_LBA_MID);
            uint8_t lba_high = inb(io + ATA_LBA_HIGH);

            if (lba_mid != 0 || lba_high != 0) {
                d->present = false;
                continue;
            }

            if (ata_poll(io) < 0) {
                d->present = false;
                continue;
            }

            uint16_t identify[256];
            for (int j = 0; j < 256; j++) {
                identify[j] = inw(io + ATA_DATA);
            }

            d->present = true;
            d->bus_width = 16;

            uint64_t lba28 = (uint64_t)(identify[61] << 16 | identify[60]);
            if (identify[83] & 0x400) {
                uint64_t lba48_low = (uint64_t)(identify[100] | (identify[101] << 16));
                uint64_t lba48_high = (uint64_t)(identify[102] | (identify[103] << 16));
                d->sectors = lba48_low | (lba48_high << 32);
            } else {
                d->sectors = lba28;
            }

            for (int j = 0; j < 40; j += 2) {
                d->model[j] = (char)(identify[27 + j/2] >> 8);
                d->model[j + 1] = (char)(identify[27 + j/2] & 0xFF);
            }
            d->model[40] = 0;

            for (int j = 0; j < 20; j += 2) {
                d->serial[j] = (char)(identify[10 + j/2] >> 8);
                d->serial[j + 1] = (char)(identify[10 + j/2] & 0xFF);
            }
            d->serial[20] = 0;

            triad_serial_puts("[ata] found drive ");
            triad_serial_puts(d->master ? "master" : "slave");
            triad_serial_puts(" on bus ");
            triad_serial_dec(bus);
            triad_serial_puts(": ");
            triad_serial_puts(d->model);
            triad_serial_puts(" sectors=");
            triad_serial_dec(d->sectors);
            triad_serial_puts("\n");
        }
    }
}

int triad_ata_identify(uint8_t bus, uint8_t drive, AtaDrive *out) {
    int idx = bus * 2 + drive;
    if (idx >= 4) return -1;
    if (!drives[idx].present) return -1;
    *out = drives[idx];
    return 0;
}

int triad_ata_read(uint8_t bus, uint8_t drive, uint64_t lba, void *buf, uint32_t sectors) {
    int idx = bus * 2 + drive;
    if (idx >= 4 || !drives[idx].present) return -1;

    uint16_t io = drives[idx].io_base;
    uint8_t drive_sel = drive == 0 ? 0xE0 : 0xF0;

    ata_wait_bsy(io);
    ata_select_drive(io, drive);

    if (lba + sectors > drives[idx].sectors) return -1;

    if (lba < 0x10000000 && sectors <= 256) {
        outb(io + ATA_DRIVE, drive_sel | ((lba >> 24) & 0x0F));
        outb(io + ATA_SECTOR_COUNT, (uint8_t)(sectors == 256 ? 0 : sectors));
        outb(io + ATA_LBA_LOW, (uint8_t)(lba & 0xFF));
        outb(io + ATA_LBA_MID, (uint8_t)((lba >> 8) & 0xFF));
        outb(io + ATA_LBA_HIGH, (uint8_t)((lba >> 16) & 0xFF));
        outb(io + ATA_COMMAND, ATA_CMD_READ);

        for (uint32_t s = 0; s < sectors; s++) {
            if (ata_poll(io) < 0) return -1;
            for (int i = 0; i < 256; i++) {
                ((uint16_t *)buf)[s * 256 + i] = inw(io + ATA_DATA);
            }
        }
    } else {
        return -1;
    }

    return 0;
}

int triad_ata_write(uint8_t bus, uint8_t drive, uint64_t lba, const void *buf, uint32_t sectors) {
    int idx = bus * 2 + drive;
    if (idx >= 4 || !drives[idx].present) return -1;

    uint16_t io = drives[idx].io_base;
    uint8_t drive_sel = drive == 0 ? 0xE0 : 0xF0;

    ata_wait_bsy(io);
    ata_select_drive(io, drive);

    if (lba + sectors > drives[idx].sectors) return -1;

    if (lba < 0x10000000 && sectors <= 256) {
        outb(io + ATA_DRIVE, drive_sel | ((lba >> 24) & 0x0F));
        outb(io + ATA_SECTOR_COUNT, (uint8_t)(sectors == 256 ? 0 : sectors));
        outb(io + ATA_LBA_LOW, (uint8_t)(lba & 0xFF));
        outb(io + ATA_LBA_MID, (uint8_t)((lba >> 8) & 0xFF));
        outb(io + ATA_LBA_HIGH, (uint8_t)((lba >> 16) & 0xFF));
        outb(io + ATA_COMMAND, ATA_CMD_WRITE);

        for (uint32_t s = 0; s < sectors; s++) {
            if (ata_poll(io) < 0) return -1;
            for (int i = 0; i < 256; i++) {
                outw(io + ATA_DATA, ((uint16_t *)buf)[s * 256 + i]);
            }
        }

        outb(io + ATA_COMMAND, ATA_CMD_FLUSH);
        ata_wait_bsy(io);
    } else {
        return -1;
    }

    return 0;
}

AtaDrive *triad_ata_get_drive(uint8_t bus, uint8_t drive) {
    int idx = bus * 2 + drive;
    if (idx >= 4) return NULL;
    if (!drives[idx].present) return NULL;
    return &drives[idx];
}

int triad_ata_read_poll(uint8_t bus, uint8_t drive, uint64_t lba, void *buf, uint32_t sectors) {
    return triad_ata_read(bus, drive, lba, buf, sectors);
}

int triad_ata_write_poll(uint8_t bus, uint8_t drive, uint64_t lba, const void *buf, uint32_t sectors) {
    return triad_ata_write(bus, drive, lba, buf, sectors);
}