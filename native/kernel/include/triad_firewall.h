#ifndef TRIAD_FIREWALL_H
#define TRIAD_FIREWALL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define FW_MAX_RULES 256
#define FW_MAX_PORTS 65536
#define FW_MAX_HOSTS 128
#define FW_MAX_PATTERNS 64

typedef enum {
    FW_ALLOW = 0,
    FW_DENY = 1,
    FW_LOG = 2,
    FW_REJECT = 3
} FirewallAction;

typedef enum {
    FW_PROTO_ANY = 0,
    FW_PROTO_TCP = 6,
    FW_PROTO_UDP = 17,
    FW_PROTO_ICMP = 1
} FirewallProto;

typedef struct {
    uint32_t src_ip;
    uint32_t src_mask;
    uint16_t src_port_start;
    uint16_t src_port_end;
    uint32_t dst_ip;
    uint32_t dst_mask;
    uint16_t dst_port_start;
    uint16_t dst_port_end;
    uint8_t protocol;
    uint8_t proc_tier;
    uint8_t action;
    uint8_t log;
    uint8_t enabled;
    uint16_t rule_id;
    char description[64];
} FirewallRule;

typedef struct {
    char pattern[128];
    uint8_t action;
    uint8_t proc_tier;
} HostPattern;

typedef struct {
    FirewallRule rules[FW_MAX_RULES];
    HostPattern host_patterns[FW_MAX_PATTERNS];
    uint32_t n_rules;
    uint32_t n_patterns;
    uint8_t default_action;
    uint8_t log_all;
    uint8_t enabled;
    uint64_t packets_allowed;
    uint64_t packets_denied;
    uint64_t bytes_allowed;
    uint64_t bytes_denied;
} FirewallState;

void triad_firewall_init(void);

int triad_firewall_add_rule(const FirewallRule *rule);
int triad_firewall_remove_rule(uint16_t rule_id);
int triad_firewall_update_rule(uint16_t rule_id, const FirewallRule *rule);
int triad_firewall_get_rule(uint16_t rule_id, FirewallRule *out);
int triad_firewall_list_rules(FirewallRule *out, int max);

int triad_firewall_add_host_pattern(const char *pattern, uint8_t action, uint8_t proc_tier);
int triad_firewall_remove_host_pattern(int idx);

bool triad_firewall_check(uint32_t src_ip, uint16_t src_port,
                          uint32_t dst_ip, uint16_t dst_port,
                          uint8_t protocol, uint8_t proc_tier);

bool triad_firewall_check_outbound(uint32_t dst_ip, uint16_t dst_port,
                                   uint8_t protocol, uint8_t proc_tier);

bool triad_firewall_check_inbound(uint32_t src_ip, uint16_t src_port,
                                  uint32_t dst_ip, uint16_t dst_port,
                                  uint8_t protocol);

int triad_firewall_set_default_action(uint8_t action);
uint8_t triad_firewall_get_default_action(void);

void triad_firewall_set_enabled(int enabled);
int triad_firewall_is_enabled(void);

void triad_firewall_set_logging(int enabled);
int triad_firewall_is_logging(void);

uint64_t triad_firewall_get_packets_allowed(void);
uint64_t triad_firewall_get_packets_denied(void);

void triad_firewall_reset_stats(void);

uint32_t triad_firewall_parse_ip(const char *str);
void triad_firewall_format_ip(uint32_t ip, char *buf, size_t buf_size);

bool triad_firewall_match_host_pattern(const char *hostname, const char *pattern);

int triad_firewall_load_default_rules(void);

#endif