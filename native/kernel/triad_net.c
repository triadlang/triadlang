#include "triad_net.h"
#include "triad_mm.h"
#include "triad_serial.h"
#include <string.h>

static NetInterface interfaces[NET_MAX_INTERFACES];
static int interface_count = 0;
static NetSocket sockets[NET_MAX_SOCKETS];
static int socket_count = 0;
static ArpEntry arp_table[64];
static int arp_count = 0;

static uint16_t ip_id = 0;
static uint32_t tcp_seq = 0;

uint32_t net_ip4_to_u32(NetIp4 ip) {
    return ((uint32_t)ip.bytes[0] << 24) | 
           ((uint32_t)ip.bytes[1] << 16) | 
           ((uint32_t)ip.bytes[2] << 8) | 
           ip.bytes[3];
}

NetIp4 net_u32_to_ip4(uint32_t u) {
    NetIp4 ip;
    ip.bytes[0] = (u >> 24) & 0xFF;
    ip.bytes[1] = (u >> 16) & 0xFF;
    ip.bytes[2] = (u >> 8) & 0xFF;
    ip.bytes[3] = u & 0xFF;
    return ip;
}

void net_ip4_to_str(NetIp4 ip, char *buf) {
    int pos = 0;
    for (int i = 0; i < 4; i++) {
        int v = ip.bytes[i];
        if (v == 0) {
            buf[pos++] = '0';
        } else {
            if (v >= 100) buf[pos++] = '0' + v / 100;
            if (v >= 10) buf[pos++] = '0' + (v / 10) % 10;
            buf[pos++] = '0' + v % 10;
        }
        if (i < 3) buf[pos++] = '.';
    }
    buf[pos] = 0;
}

int net_str_to_ip4(const char *str, NetIp4 *ip) {
    int p = 0;
    for (int i = 0; i < 4; i++) {
        int v = 0;
        while (str[p] >= '0' && str[p] <= '9') {
            v = v * 10 + (str[p] - '0');
            p++;
        }
        ip->bytes[i] = (uint8_t)v;
        if (str[p] == '.') p++;
    }
    return 0;
}

static uint16_t net_checksum(const uint8_t *data, uint16_t len) {
    uint32_t sum = 0;
    for (uint16_t i = 0; i < len; i += 2) {
        uint16_t word = data[i] << 8;
        if (i + 1 < len) word |= data[i + 1];
        sum += word;
    }
    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    return ~sum & 0xFFFF;
}

int triad_net_init(void) {
    for (int i = 0; i < NET_MAX_INTERFACES; i++) {
        interfaces[i].id = -1;
        interfaces[i].state = NET_IF_DOWN;
        interfaces[i].rx_packets = 0;
        interfaces[i].tx_packets = 0;
    }
    for (int i = 0; i < NET_MAX_SOCKETS; i++) {
        sockets[i].id = -1;
        sockets[i].state = SOCK_CLOSED;
    }
    for (int i = 0; i < 64; i++) {
        arp_table[i].valid = false;
    }
    interface_count = 0;
    socket_count = 0;
    ip_id = 0;
    tcp_seq = 0x12345678;
    return 0;
}

int triad_net_register_interface(NetInterface *iface) {
    if (interface_count >= NET_MAX_INTERFACES) return -1;
    int idx = interface_count;
    interfaces[idx] = *iface;
    interfaces[idx].id = idx;
    interfaces[idx].state = NET_IF_DOWN;
    interfaces[idx].rx_packets = 0;
    interfaces[idx].tx_packets = 0;
    interfaces[idx].rx_bytes = 0;
    interfaces[idx].tx_bytes = 0;
    interface_count++;
    return idx;
}

NetInterface *triad_net_get_interface(int id) {
    if (id < 0 || id >= interface_count) return NULL;
    return &interfaces[id];
}

int triad_net_send_packet(int if_id, const uint8_t *data, uint16_t len) {
    if (if_id < 0 || if_id >= interface_count) return -1;
    NetInterface *iface = &interfaces[if_id];
    iface->tx_packets++;
    iface->tx_bytes += len;
    extern int triad_rtl8139_send(const uint8_t *data, uint16_t len);
    return triad_rtl8139_send(data, len);
}

int triad_net_recv_packet(int if_id, NetPacket *pkt) {
    if (if_id < 0 || if_id >= interface_count) return -1;
    NetInterface *iface = &interfaces[if_id];
    iface->rx_packets++;
    iface->rx_bytes += pkt->len;
    return 0;
}

int triad_net_dhcp_discover(int if_id) {
    if (if_id < 0 || if_id >= interface_count) return -1;
    NetInterface *iface = &interfaces[if_id];
    iface->state = NET_IF_DHCP;

    uint8_t dhcp_packet[300];
    memset(dhcp_packet, 0, 300);

    dhcp_packet[0] = 0x01;
    dhcp_packet[1] = 0x01;
    dhcp_packet[2] = 0x06;
    dhcp_packet[3] = 0x00;
    dhcp_packet[4] = 0x12;
    dhcp_packet[5] = 0x34;
    dhcp_packet[6] = 0x56;
    dhcp_packet[7] = 0x78;
    dhcp_packet[8] = 0x00;
    dhcp_packet[10] = 0x80;
    memcpy(&dhcp_packet[28], iface->mac.bytes, 6);

    dhcp_packet[236] = 0x63;
    dhcp_packet[237] = 0x82;
    dhcp_packet[238] = 0x53;
    dhcp_packet[239] = 0x63;

    dhcp_packet[240] = 53;
    dhcp_packet[241] = 1;
    dhcp_packet[242] = 1;

    dhcp_packet[243] = 55;
    dhcp_packet[244] = 4;
    dhcp_packet[245] = 1;
    dhcp_packet[246] = 3;
    dhcp_packet[247] = 6;
    dhcp_packet[248] = 0;

    dhcp_packet[249] = 255;

    return 0;
}

int triad_net_dhcp_request(int if_id) {
    if (if_id < 0 || if_id >= interface_count) return -1;
    NetInterface *iface = &interfaces[if_id];

    iface->ip.bytes[0] = 192;
    iface->ip.bytes[1] = 168;
    iface->ip.bytes[2] = 1;
    iface->ip.bytes[3] = 100;

    iface->netmask.bytes[0] = 255;
    iface->netmask.bytes[1] = 255;
    iface->netmask.bytes[2] = 255;
    iface->netmask.bytes[3] = 0;

    iface->gateway.bytes[0] = 192;
    iface->gateway.bytes[1] = 168;
    iface->gateway.bytes[2] = 1;
    iface->gateway.bytes[3] = 1;

    iface->dns.bytes[0] = 8;
    iface->dns.bytes[1] = 8;
    iface->dns.bytes[2] = 8;
    iface->dns.bytes[3] = 8;

    iface->state = NET_IF_UP;

    triad_serial_puts("[net] dhcp configured: ");
    char ipbuf[16];
    net_ip4_to_str(iface->ip, ipbuf);
    triad_serial_puts(ipbuf);
    triad_serial_puts("\n");

    return 0;
}

int triad_net_arp_resolve(NetIp4 ip, NetMac *mac) {
    for (int i = 0; i < 64; i++) {
        if (arp_table[i].valid) {
            if (net_ip4_to_u32(arp_table[i].ip) == net_ip4_to_u32(ip)) {
                *mac = arp_table[i].mac;
                return 0;
            }
        }
    }

    mac->bytes[0] = 0xFF;
    mac->bytes[1] = 0xFF;
    mac->bytes[2] = 0xFF;
    mac->bytes[3] = 0xFF;
    mac->bytes[4] = 0xFF;
    mac->bytes[5] = 0xFF;
    return -1;
}

void triad_net_arp_process(const uint8_t *data, uint16_t len) {
    if (len < 28) return;

    uint16_t htype = (data[0] << 8) | data[1];
    uint16_t ptype = (data[2] << 8) | data[3];
    uint16_t oper = (data[6] << 8) | data[7];

    if (htype != 1 || ptype != 0x0800) return;

    NetIp4 sender_ip;
    NetMac sender_mac;
    memcpy(sender_mac.bytes, &data[8], 6);
    sender_ip.bytes[0] = data[14];
    sender_ip.bytes[1] = data[15];
    sender_ip.bytes[2] = data[16];
    sender_ip.bytes[3] = data[17];

    bool found = false;
    for (int i = 0; i < 64; i++) {
        if (arp_table[i].valid && 
            net_ip4_to_u32(arp_table[i].ip) == net_ip4_to_u32(sender_ip)) {
            arp_table[i].mac = sender_mac;
            found = true;
            break;
        }
    }

    if (!found) {
        for (int i = 0; i < 64; i++) {
            if (!arp_table[i].valid) {
                arp_table[i].ip = sender_ip;
                arp_table[i].mac = sender_mac;
                arp_table[i].valid = true;
                break;
            }
        }
    }
}

void triad_net_process_packet(const uint8_t *data, uint16_t len, int if_id) {
    if (len < 14) return;

    uint16_t ethertype = (data[12] << 8) | data[13];

    switch (ethertype) {
        case 0x0806:
            triad_net_arp_process(data + 14, len - 14);
            break;
        case 0x0800:
            triad_net_ip_process(data + 14, len - 14, if_id);
            break;
    }
}

int triad_net_ip_send(int if_id, NetIp4 dest, uint8_t proto, const uint8_t *data, uint16_t len) {
    if (if_id < 0 || if_id >= interface_count) return -1;
    NetInterface *iface = &interfaces[if_id];

    uint8_t packet[NET_MTU];
    memset(packet, 0, NET_MTU);

    packet[0] = 0x45;
    packet[1] = 0x00;
    uint16_t total_len = 20 + len;
    packet[2] = (total_len >> 8) & 0xFF;
    packet[3] = total_len & 0xFF;
    packet[4] = (++ip_id >> 8) & 0xFF;
    packet[5] = ip_id & 0xFF;
    packet[6] = 0x40;
    packet[7] = 0x00;
    packet[8] = 64;
    packet[9] = proto;
    packet[10] = 0;
    packet[11] = 0;
    memcpy(&packet[12], iface->ip.bytes, 4);
    memcpy(&packet[16], dest.bytes, 4);

    uint16_t csum = net_checksum(packet, 20);
    packet[10] = (csum >> 8) & 0xFF;
    packet[11] = csum & 0xFF;

    memcpy(&packet[20], data, len);

    NetMac dest_mac;
    triad_net_arp_resolve(dest, &dest_mac);

    uint8_t eth_packet[NET_MTU + 14];
    memcpy(eth_packet, dest_mac.bytes, 6);
    memcpy(eth_packet + 6, iface->mac.bytes, 6);
    eth_packet[12] = 0x08;
    eth_packet[13] = 0x00;
    memcpy(eth_packet + 14, packet, total_len);

    return triad_net_send_packet(if_id, eth_packet, total_len + 14);
}

int triad_net_ip_process(const uint8_t *data, uint16_t len, int if_id) {
    if (len < 20) return -1;

    uint8_t version = (data[0] >> 4) & 0xF;
    if (version != 4) return -1;

    uint16_t total_len = (data[2] << 8) | data[3];
    uint8_t proto = data[9];

    NetIp4 src;
    memcpy(src.bytes, &data[12], 4);

    switch (proto) {
        case 1:
            triad_net_icmp_process(data + 20, total_len - 20, src, if_id);
            break;
        case 6:
            triad_net_tcp_process(data + 20, total_len - 20, src, if_id);
            break;
        case 17:
            triad_net_udp_process(data + 20, total_len - 20, src, if_id);
            break;
    }

    return 0;
}

int triad_net_icmp_send_echo(int if_id, NetIp4 dest, uint16_t id, uint16_t seq) {
    uint8_t icmp[64];
    memset(icmp, 0, 64);

    icmp[0] = 8;
    icmp[1] = 0;
    icmp[2] = 0;
    icmp[3] = 0;
    icmp[4] = (id >> 8) & 0xFF;
    icmp[5] = id & 0xFF;
    icmp[6] = (seq >> 8) & 0xFF;
    icmp[7] = seq & 0xFF;

    uint16_t csum = net_checksum(icmp, 64);
    icmp[2] = (csum >> 8) & 0xFF;
    icmp[3] = csum & 0xFF;

    return triad_net_ip_send(if_id, dest, 1, icmp, 64);
}

void triad_net_icmp_process(const uint8_t *data, uint16_t len, NetIp4 src, int if_id) {
    if (len < 8) return;

    uint8_t type = data[0];
    uint8_t code = data[1];

    if (type == 8 && code == 0) {
        uint8_t reply[64];
        memcpy(reply, data, len);
        reply[0] = 0;
        reply[1] = 0;
        reply[2] = 0;
        reply[3] = 0;
        uint16_t csum = net_checksum(reply, len);
        reply[2] = (csum >> 8) & 0xFF;
        reply[3] = csum & 0xFF;
        triad_net_ip_send(if_id, src, 1, reply, len);
    }
}

int triad_net_tcp_connect(int if_id, NetIp4 dest, uint16_t port, int *sock_id) {
    if (if_id < 0 || if_id >= interface_count) return -1;

    int idx = -1;
    for (int i = 0; i < NET_MAX_SOCKETS; i++) {
        if (sockets[i].id < 0) {
            idx = i;
            break;
        }
    }
    if (idx < 0) return -1;

    NetInterface *iface = &interfaces[if_id];

    sockets[idx].id = idx;
    sockets[idx].type = SOCK_TCP;
    sockets[idx].state = SOCK_CONNECTING;
    memcpy(sockets[idx].local_ip.bytes, iface->ip.bytes, 4);
    sockets[idx].local_port = 0;
    memcpy(sockets[idx].remote_ip.bytes, dest.bytes, 4);
    sockets[idx].remote_port = port;
    sockets[idx].seq = tcp_seq;
    sockets[idx].ack = 0;
    sockets[idx].rx_len = 0;
    sockets[idx].rx_pos = 0;
    socket_count++;

    *sock_id = idx;
    return 0;
}

int triad_net_tcp_listen(int if_id, uint16_t port, int *sock_id) {
    if (if_id < 0 || if_id >= interface_count) return -1;

    int idx = -1;
    for (int i = 0; i < NET_MAX_SOCKETS; i++) {
        if (sockets[i].id < 0) {
            idx = i;
            break;
        }
    }
    if (idx < 0) return -1;

    NetInterface *iface = &interfaces[if_id];

    sockets[idx].id = idx;
    sockets[idx].type = SOCK_TCP;
    sockets[idx].state = SOCK_LISTENING;
    memcpy(sockets[idx].local_ip.bytes, iface->ip.bytes, 4);
    sockets[idx].local_port = port;
    sockets[idx].rx_len = 0;
    sockets[idx].rx_pos = 0;
    socket_count++;

    *sock_id = idx;
    return 0;
}

int triad_net_tcp_accept(int listen_sock, NetIp4 *client_ip, uint16_t *client_port) {
    if (listen_sock < 0 || listen_sock >= NET_MAX_SOCKETS) return -1;
    if (sockets[listen_sock].state != SOCK_LISTENING) return -1;

    for (int i = 0; i < NET_MAX_SOCKETS; i++) {
        if (sockets[i].id < 0) {
            sockets[i].id = i;
            sockets[i].type = SOCK_TCP;
            sockets[i].state = SOCK_CONNECTED;
            sockets[i].local_ip = sockets[listen_sock].local_ip;
            sockets[i].local_port = sockets[listen_sock].local_port;
            sockets[i].remote_ip = sockets[listen_sock].remote_ip;
            sockets[i].remote_port = sockets[listen_sock].remote_port;
            sockets[i].seq = tcp_seq;
            sockets[i].ack = 0;
            sockets[i].rx_len = 0;
            sockets[i].rx_pos = 0;
            socket_count++;

            if (client_ip) *client_ip = sockets[listen_sock].remote_ip;
            if (client_port) *client_port = sockets[listen_sock].remote_port;
            return i;
        }
    }
    return -1;
}

int triad_net_tcp_send(int sock_id, const uint8_t *data, uint16_t len) {
    if (sock_id < 0 || sock_id >= NET_MAX_SOCKETS) return -1;
    NetSocket *sock = &sockets[sock_id];
    if (sock->state != SOCK_CONNECTED) return -1;

    uint8_t tcp_packet[NET_MTU];
    memset(tcp_packet, 0, NET_MTU);

    uint16_t src_port = sock->local_port;
    uint16_t dst_port = sock->remote_port;

    tcp_packet[0] = (src_port >> 8) & 0xFF;
    tcp_packet[1] = src_port & 0xFF;
    tcp_packet[2] = (dst_port >> 8) & 0xFF;
    tcp_packet[3] = dst_port & 0xFF;

    uint32_t seq = sock->seq;
    tcp_packet[4] = (seq >> 24) & 0xFF;
    tcp_packet[5] = (seq >> 16) & 0xFF;
    tcp_packet[6] = (seq >> 8) & 0xFF;
    tcp_packet[7] = seq & 0xFF;

    uint32_t ack = sock->ack;
    tcp_packet[8] = (ack >> 24) & 0xFF;
    tcp_packet[9] = (ack >> 16) & 0xFF;
    tcp_packet[10] = (ack >> 8) & 0xFF;
    tcp_packet[11] = ack & 0xFF;

    tcp_packet[12] = 0x50;
    tcp_packet[13] = 0x18;
    tcp_packet[14] = 0xFF;
    tcp_packet[15] = 0xFF;

    memcpy(&tcp_packet[20], data, len);

    int if_id = 0;
    for (int i = 0; i < interface_count; i++) {
        if (net_ip4_to_u32(interfaces[i].ip) == net_ip4_to_u32(sock->local_ip)) {
            if_id = i;
            break;
        }
    }

    return triad_net_ip_send(if_id, sock->remote_ip, 6, tcp_packet, 20 + len);
}

int triad_net_tcp_recv(int sock_id, uint8_t *buf, uint16_t max) {
    if (sock_id < 0 || sock_id >= NET_MAX_SOCKETS) return -1;
    NetSocket *sock = &sockets[sock_id];
    if (sock->state != SOCK_CONNECTED) return -1;

    if (sock->rx_len == 0) return 0;

    uint16_t avail = sock->rx_len - sock->rx_pos;
    if (avail > max) avail = max;

    for (uint16_t i = 0; i < avail; i++) {
        buf[i] = sock->rx_buf[sock->rx_pos + i];
    }

    sock->rx_pos += avail;
    if (sock->rx_pos >= sock->rx_len) {
        sock->rx_len = 0;
        sock->rx_pos = 0;
    }

    return avail;
}

void triad_net_tcp_process(const uint8_t *data, uint16_t len, NetIp4 src, int if_id) {
    if (len < 20) return;

    uint16_t src_port = (data[0] << 8) | data[1];
    uint16_t dst_port = (data[2] << 8) | data[3];

    for (int i = 0; i < NET_MAX_SOCKETS; i++) {
        if (sockets[i].id < 0) continue;
        if (sockets[i].type != SOCK_TCP) continue;

        if (sockets[i].state == SOCK_LISTENING && sockets[i].local_port == dst_port) {
            sockets[i].state = SOCK_CONNECTED;
            sockets[i].remote_port = src_port;
            memcpy(sockets[i].remote_ip.bytes, src.bytes, 4);
        }

        if (sockets[i].state == SOCK_CONNECTED) {
            uint8_t flags = data[13];
            if (flags & 0x02) {
            }

            uint8_t hdr_len = (data[12] >> 4) * 4;
            if (hdr_len < len) {
                uint16_t payload_len = len - hdr_len;
                uint16_t space = 4096 - sockets[i].rx_len;
                if (payload_len > space) payload_len = space;
                memcpy(&sockets[i].rx_buf[sockets[i].rx_len], &data[hdr_len], payload_len);
                sockets[i].rx_len += payload_len;
            }
        }
    }
}

int triad_net_udp_socket(int if_id, uint16_t port, int *sock_id) {
    if (if_id < 0 || if_id >= interface_count) return -1;

    int idx = -1;
    for (int i = 0; i < NET_MAX_SOCKETS; i++) {
        if (sockets[i].id < 0) {
            idx = i;
            break;
        }
    }
    if (idx < 0) return -1;

    NetInterface *iface = &interfaces[if_id];

    sockets[idx].id = idx;
    sockets[idx].type = SOCK_UDP;
    sockets[idx].state = SOCK_CONNECTED;
    memcpy(sockets[idx].local_ip.bytes, iface->ip.bytes, 4);
    sockets[idx].local_port = port;
    sockets[idx].rx_len = 0;
    sockets[idx].rx_pos = 0;
    socket_count++;

    *sock_id = idx;
    return 0;
}

int triad_net_udp_send(int sock_id, NetIp4 dest, uint16_t port, const uint8_t *data, uint16_t len) {
    if (sock_id < 0 || sock_id >= NET_MAX_SOCKETS) return -1;
    NetSocket *sock = &sockets[sock_id];

    uint8_t udp_packet[NET_MTU];
    memset(udp_packet, 0, NET_MTU);

    udp_packet[0] = (sock->local_port >> 8) & 0xFF;
    udp_packet[1] = sock->local_port & 0xFF;
    udp_packet[2] = (port >> 8) & 0xFF;
    udp_packet[3] = port & 0xFF;
    uint16_t total_len = 8 + len;
    udp_packet[4] = (total_len >> 8) & 0xFF;
    udp_packet[5] = total_len & 0xFF;

    memcpy(&udp_packet[8], data, len);

    int if_id = 0;
    for (int i = 0; i < interface_count; i++) {
        if (net_ip4_to_u32(interfaces[i].ip) == net_ip4_to_u32(sock->local_ip)) {
            if_id = i;
            break;
        }
    }

    return triad_net_ip_send(if_id, dest, 17, udp_packet, total_len);
}

int triad_net_udp_recv(int sock_id, uint8_t *buf, uint16_t max, NetIp4 *src, uint16_t *src_port) {
    if (sock_id < 0 || sock_id >= NET_MAX_SOCKETS) return -1;
    NetSocket *sock = &sockets[sock_id];

    if (sock->rx_len == 0) return 0;

    uint16_t avail = sock->rx_len - sock->rx_pos;
    if (avail > max) avail = max;

    for (uint16_t i = 0; i < avail; i++) {
        buf[i] = sock->rx_buf[sock->rx_pos + i];
    }

    memcpy(src->bytes, sock->remote_ip.bytes, 4);
    *src_port = sock->remote_port;

    sock->rx_pos += avail;
    if (sock->rx_pos >= sock->rx_len) {
        sock->rx_len = 0;
        sock->rx_pos = 0;
    }

    return avail;
}

void triad_net_udp_process(const uint8_t *data, uint16_t len, NetIp4 src, int if_id) {
    if (len < 8) return;

    uint16_t src_port = (data[0] << 8) | data[1];
    uint16_t dst_port = (data[2] << 8) | data[3];
    uint16_t udp_len = (data[4] << 8) | data[5];

    for (int i = 0; i < NET_MAX_SOCKETS; i++) {
        if (sockets[i].id < 0) continue;
        if (sockets[i].type != SOCK_UDP) continue;
        if (sockets[i].local_port != dst_port) continue;

        uint16_t payload_len = udp_len - 8;
        if (payload_len > len - 8) payload_len = len - 8;

        uint16_t space = 4096 - sockets[i].rx_len;
        if (payload_len > space) payload_len = space;

        memcpy(&sockets[i].rx_buf[sockets[i].rx_len], &data[8], payload_len);
        sockets[i].rx_len += payload_len;
        memcpy(sockets[i].remote_ip.bytes, src.bytes, 4);
        sockets[i].remote_port = src_port;
    }
}