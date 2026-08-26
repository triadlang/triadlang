#ifndef TRIAD_ATA_H
#define TRIAD_ATA_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define ATA_PRIMARY_IO    0x1F0
#define ATA_PRIMARY_CTRL   0x3F6
#define ATA_SECONDARY_IO   0x170
#define ATA_SECONDARY_CTRL  0x376

#define ATA_DATA        0
#define ATA_ERROR        1
#define ATA_FEATURES     1
#define ATA_SECTOR_COUNT 2
#define ATA_LBA_LOW      3
#define ATA_LBA_MID      4
#define ATA_LBA_HIGH     5
#define ATA_DRIVE        6
#define ATA_STATUS       7
#define ATA_COMMAND      7

#define ATA_CMD_READ     0x20
#define ATA_CMD_WRITE     0x30
#define ATA_CMD_IDENTIFY 0xEC
#define ATA_CMD_FLUSH     0xE7

#define ATA_STATUS_BSY   0x80
#define ATA_STATUS_DRDY  0x40
#define ATA_STATUS_DF    0x20
#define ATA_STATUS_DRQ   0x08
#define ATA_STATUS_ERR   0x01

#define ATA_MASTER  0x00
#define ATA_SLAVE   0x01

typedef struct AtaDrive {
    bool present;
    bool master;
    uint16_t io_base;
    uint16_t ctrl_base;
    uint64_t sectors;
    char model[41];
    char serial[21];
    uint8_t bus_width;
} AtaDrive;

void triad_ata_init(void);
int triad_ata_identify(uint8_t bus, uint8_t drive, AtaDrive *out);
int triad_ata_read(uint8_t bus, uint8_t drive, uint64_t lba, void *buf, uint32_t sectors);
int triad_ata_write(uint8_t bus, uint8_t drive, uint64_t lba, const void *buf, uint32_t sectors);
int triad_ata_read_poll(uint8_t bus, uint8_t drive, uint64_t lba, void *buf, uint32_t sectors);
int triad_ata_write_poll(uint8_t bus, uint8_t drive, uint64_t lba, const void *buf, uint32_t sectors);

AtaDrive *triad_ata_get_drive(uint8_t bus, uint8_t drive);

#endif