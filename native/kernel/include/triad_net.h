#ifndef TRIAD_NET_H
#define TRIAD_NET_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define NET_MAX_INTERFACES  4
#define NET_MAX_SOCKETS     32
#define NET_RX_RING_SIZE    256
#define NET_TX_RING_SIZE    256
#define NET_MTU             1500
#define NET_MAC_LEN         6

typedef struct NetMac {
    uint8_t bytes[NET_MAC_LEN];
} NetMac;

typedef struct NetIp4 {
    uint8_t bytes[4];
} NetIp4;

typedef enum {
    NET_IF_DOWN = 0,
    NET_IF_UP = 1,
    NET_IF_DHCP = 2
} NetIfState;

typedef struct NetInterface {
    int id;
    char name[16];
    NetMac mac;
    NetIp4 ip;
    NetIp4 netmask;
    NetIp4 gateway;
    NetIp4 dns;
    NetIfState state;
    uint64_t rx_packets;
    uint64_t tx_packets;
    uint64_t rx_bytes;
    uint64_t tx_bytes;
    void *driver_data;
} NetInterface;

typedef struct NetPacket {
    uint8_t data[NET_MTU + 14];
    uint16_t len;
    int if_id;
} NetPacket;

typedef enum {
    SOCK_TCP = 1,
    SOCK_UDP = 2,
    SOCK_RAW = 3
} NetSocketType;

typedef enum {
    SOCK_CLOSED = 0,
    SOCK_LISTENING = 1,
    SOCK_CONNECTING = 2,
    SOCK_CONNECTED = 3,
    SOCK_CLOSING = 4
} NetSocketState;

typedef struct NetSocket {
    int id;
    NetSocketType type;
    NetSocketState state;
    NetIp4 local_ip;
    uint16_t local_port;
    NetIp4 remote_ip;
    uint16_t remote_port;
    uint32_t seq;
    uint32_t ack;
    uint8_t rx_buf[4096];
    uint16_t rx_len;
    uint16_t rx_pos;
} NetSocket;

typedef struct ArpEntry {
    NetIp4 ip;
    NetMac mac;
    uint64_t timestamp;
    bool valid;
} ArpEntry;

int triad_net_init(void);
int triad_net_register_interface(NetInterface *iface);
NetInterface *triad_net_get_interface(int id);

int triad_net_send_packet(int if_id, const uint8_t *data, uint16_t len);
int triad_net_recv_packet(int if_id, NetPacket *pkt);

int triad_net_dhcp_discover(int if_id);
int triad_net_dhcp_request(int if_id);

int triad_net_arp_resolve(NetIp4 ip, NetMac *mac);
void triad_net_arp_process(const uint8_t *data, uint16_t len);

void triad_net_process_packet(const uint8_t *data, uint16_t len, int if_id);

int triad_net_ip_send(int if_id, NetIp4 dest, uint8_t proto, const uint8_t *data, uint16_t len);
int triad_net_ip_process(const uint8_t *data, uint16_t len, int if_id);

int triad_net_icmp_send_echo(int if_id, NetIp4 dest, uint16_t id, uint16_t seq);
void triad_net_icmp_process(const uint8_t *data, uint16_t len, NetIp4 src, int if_id);

int triad_net_tcp_connect(int if_id, NetIp4 dest, uint16_t port, int *sock_id);
int triad_net_tcp_listen(int if_id, uint16_t port, int *sock_id);
int triad_net_tcp_accept(int listen_sock, NetIp4 *client_ip, uint16_t *client_port);
int triad_net_tcp_send(int sock_id, const uint8_t *data, uint16_t len);
int triad_net_tcp_recv(int sock_id, uint8_t *buf, uint16_t max);
void triad_net_tcp_process(const uint8_t *data, uint16_t len, NetIp4 src, int if_id);

int triad_net_udp_socket(int if_id, uint16_t port, int *sock_id);
int triad_net_udp_send(int sock_id, NetIp4 dest, uint16_t port, const uint8_t *data, uint16_t len);
int triad_net_udp_recv(int sock_id, uint8_t *buf, uint16_t max, NetIp4 *src, uint16_t *src_port);
void triad_net_udp_process(const uint8_t *data, uint16_t len, NetIp4 src, int if_id);

uint32_t net_ip4_to_u32(NetIp4 ip);
NetIp4 net_u32_to_ip4(uint32_t u);
void net_ip4_to_str(NetIp4 ip, char *buf);
int net_str_to_ip4(const char *str, NetIp4 *ip);

#endif