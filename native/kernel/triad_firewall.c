#include "triad_firewall.h"
#include "triad_kernel.h"
#include "triad_audit.h"
#include "triad_serial.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static FirewallState fw_state = {0};
static int fw_initialized = 0;

static int match_ip(uint32_t ip, uint32_t net, uint32_t mask) {
    return (ip & mask) == (net & mask);
}

static int match_port(uint16_t port, uint16_t start, uint16_t end) {
    return port >= start && port <= end;
}

void triad_firewall_init(void) {
    if (fw_initialized) return;

    fw_state.n_rules = 0;
    fw_state.n_patterns = 0;
    fw_state.default_action = FW_DENY;
    fw_state.log_all = 0;
    fw_state.enabled = 1;
    fw_state.packets_allowed = 0;
    fw_state.packets_denied = 0;
    fw_state.bytes_allowed = 0;
    fw_state.bytes_denied = 0;

    for (int i = 0; i < FW_MAX_RULES; i++) {
        fw_state.rules[i].enabled = 0;
        fw_state.rules[i].rule_id = 0;
    }

    for (int i = 0; i < FW_MAX_PATTERNS; i++) {
        fw_state.host_patterns[i].pattern[0] = 0;
        fw_state.host_patterns[i].action = FW_DENY;
        fw_state.host_patterns[i].proc_tier = PROC_T3_USER;
    }

    triad_firewall_load_default_rules();
    fw_initialized = 1;
}

int triad_firewall_add_rule(const FirewallRule *rule) {
    if (!rule) return -1;
    if (fw_state.n_rules >= FW_MAX_RULES) return -1;

    uint16_t new_id = 1;
    for (int i = 0; i < (int)fw_state.n_rules; i++) {
        if (fw_state.rules[i].rule_id >= new_id) {
            new_id = (uint16_t)(fw_state.rules[i].rule_id + 1);
        }
    }
    if (new_id == 0) new_id = 1;

    for (int i = 0; i < (int)fw_state.n_rules; i++) {
        if (!fw_state.rules[i].enabled) {
            fw_state.rules[i] = *rule;
            fw_state.rules[i].enabled = 1;
            fw_state.rules[i].rule_id = new_id;
            triad_audit_log(AUDIT_NET_CONNECT, AUDIT_SEV_INFO,
                            "firewall rule added: id=%d action=%d",
                            new_id, rule->action);
            return new_id;
        }
    }

    fw_state.rules[fw_state.n_rules] = *rule;
    fw_state.rules[fw_state.n_rules].enabled = 1;
    fw_state.rules[fw_state.n_rules].rule_id = new_id;
    fw_state.n_rules++;

    triad_audit_log(AUDIT_NET_CONNECT, AUDIT_SEV_INFO,
                    "firewall rule added: id=%d action=%d",
                    new_id, rule->action);

    return new_id;
}

int triad_firewall_remove_rule(uint16_t rule_id) {
    for (int i = 0; i < (int)fw_state.n_rules; i++) {
        if (fw_state.rules[i].rule_id == rule_id && fw_state.rules[i].enabled) {
            fw_state.rules[i].enabled = 0;
            triad_audit_log(AUDIT_NET_CONNECT, AUDIT_SEV_INFO,
                            "firewall rule removed: id=%d", rule_id);
            return 0;
        }
    }
    return -1;
}

int triad_firewall_update_rule(uint16_t rule_id, const FirewallRule *rule) {
    for (int i = 0; i < (int)fw_state.n_rules; i++) {
        if (fw_state.rules[i].rule_id == rule_id && fw_state.rules[i].enabled) {
            uint16_t id = fw_state.rules[i].rule_id;
            fw_state.rules[i] = *rule;
            fw_state.rules[i].rule_id = id;
            fw_state.rules[i].enabled = 1;
            triad_audit_log(AUDIT_NET_CONNECT, AUDIT_SEV_INFO,
                            "firewall rule updated: id=%d", rule_id);
            return 0;
        }
    }
    return -1;
}

int triad_firewall_get_rule(uint16_t rule_id, FirewallRule *out) {
    if (!out) return -1;
    for (int i = 0; i < (int)fw_state.n_rules; i++) {
        if (fw_state.rules[i].rule_id == rule_id && fw_state.rules[i].enabled) {
            *out = fw_state.rules[i];
            return 0;
        }
    }
    return -1;
}

int triad_firewall_list_rules(FirewallRule *out, int max) {
    if (!out || max <= 0) return 0;
    int count = 0;
    for (int i = 0; i < (int)fw_state.n_rules && count < max; i++) {
        if (fw_state.rules[i].enabled) {
            out[count++] = fw_state.rules[i];
        }
    }
    return count;
}

int triad_firewall_add_host_pattern(const char *pattern, uint8_t action, uint8_t proc_tier) {
    if (!pattern || fw_state.n_patterns >= FW_MAX_PATTERNS) return -1;

    int idx = fw_state.n_patterns;
    int k = 0;
    while (pattern[k] && k < 127) {
        fw_state.host_patterns[idx].pattern[k] = pattern[k];
        k++;
    }
    fw_state.host_patterns[idx].pattern[k] = 0;
    fw_state.host_patterns[idx].action = action;
    fw_state.host_patterns[idx].proc_tier = proc_tier;
    fw_state.n_patterns++;

    return idx;
}

int triad_firewall_remove_host_pattern(int idx) {
    if (idx < 0 || idx >= (int)fw_state.n_patterns) return -1;
    fw_state.host_patterns[idx].pattern[0] = 0;
    return 0;
}

static FirewallAction triad_firewall_check_internal(uint32_t src_ip, uint16_t src_port,
                                                     uint32_t dst_ip, uint16_t dst_port,
                                                     uint8_t protocol, uint8_t proc_tier) {
    if (!fw_state.enabled) return FW_ALLOW;

    for (int i = 0; i < (int)fw_state.n_rules; i++) {
        FirewallRule *r = &fw_state.rules[i];
        if (!r->enabled) continue;

        if (r->proc_tier != 0xFF && r->proc_tier != proc_tier) continue;

        if (r->src_ip != 0 && !match_ip(src_ip, r->src_ip, r->src_mask)) continue;

        if (r->src_port_start != 0 && !match_port(src_port, r->src_port_start, r->src_port_end)) continue;

        if (r->dst_ip != 0 && !match_ip(dst_ip, r->dst_ip, r->dst_mask)) continue;

        if (r->dst_port_start != 0 && !match_port(dst_port, r->dst_port_start, r->dst_port_end)) continue;

        if (r->protocol != FW_PROTO_ANY && r->protocol != protocol) continue;

        return (FirewallAction)r->action;
    }

    return (FirewallAction)fw_state.default_action;
}

bool triad_firewall_check(uint32_t src_ip, uint16_t src_port,
                          uint32_t dst_ip, uint16_t dst_port,
                          uint8_t protocol, uint8_t proc_tier) {
    FirewallAction action = triad_firewall_check_internal(src_ip, src_port, dst_ip, dst_port, protocol, proc_tier);

    if (action == FW_ALLOW) {
        fw_state.packets_allowed++;
        return true;
    } else if (action == FW_LOG) {
        fw_state.packets_allowed++;
        triad_audit_log(AUDIT_NET_CONNECT, AUDIT_SEV_INFO,
                        "firewall: allowed logged src=%x dst=%x:%d proto=%d tier=%d",
                        src_ip, dst_ip, dst_port, protocol, proc_tier);
        return true;
    } else {
        fw_state.packets_denied++;
        triad_audit_log(AUDIT_NET_CONNECT, AUDIT_SEV_WARN,
                        "firewall: denied src=%x dst=%x:%d proto=%d tier=%d",
                        src_ip, dst_ip, dst_port, protocol, proc_tier);
        return false;
    }
}

bool triad_firewall_check_outbound(uint32_t dst_ip, uint16_t dst_port,
                                   uint8_t protocol, uint8_t proc_tier) {
    return triad_firewall_check(0, 0, dst_ip, dst_port, protocol, proc_tier);
}

bool triad_firewall_check_inbound(uint32_t src_ip, uint16_t src_port,
                                  uint32_t dst_ip, uint16_t dst_port,
                                  uint8_t protocol) {
    return triad_firewall_check(src_ip, src_port, dst_ip, dst_port, protocol, 0xFF);
}

int triad_firewall_set_default_action(uint8_t action) {
    if (action > FW_REJECT) return -1;
    fw_state.default_action = action;
    return 0;
}

uint8_t triad_firewall_get_default_action(void) {
    return fw_state.default_action;
}

void triad_firewall_set_enabled(int enabled) {
    fw_state.enabled = enabled ? 1 : 0;
}

int triad_firewall_is_enabled(void) {
    return fw_state.enabled;
}

void triad_firewall_set_logging(int enabled) {
    fw_state.log_all = enabled ? 1 : 0;
}

int triad_firewall_is_logging(void) {
    return fw_state.log_all;
}

uint64_t triad_firewall_get_packets_allowed(void) {
    return fw_state.packets_allowed;
}

uint64_t triad_firewall_get_packets_denied(void) {
    return fw_state.packets_denied;
}

void triad_firewall_reset_stats(void) {
    fw_state.packets_allowed = 0;
    fw_state.packets_denied = 0;
    fw_state.bytes_allowed = 0;
    fw_state.bytes_denied = 0;
}

uint32_t triad_firewall_parse_ip(const char *str) {
    if (!str) return 0;

    uint32_t ip = 0;
    uint32_t parts[4] = {0};
    int part_idx = 0;
    uint32_t val = 0;

    while (*str && part_idx < 4) {
        if (*str >= '0' && *str <= '9') {
            val = val * 10 + (*str - '0');
        } else if (*str == '.') {
            parts[part_idx++] = val;
            val = 0;
        } else {
            break;
        }
        str++;
    }
    if (part_idx < 4) {
        parts[part_idx] = val;
    }

    ip = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3];
    return ip;
}

void triad_firewall_format_ip(uint32_t ip, char *buf, size_t buf_size) {
    if (!buf || buf_size < 16) return;

    int written = 0;
    for (int i = 3; i >= 0; i--) {
        uint8_t byte = (ip >> (i * 8)) & 0xFF;
        if (byte >= 100) {
            if (written < (int)buf_size - 1) buf[written++] = '0' + (byte / 100);
        }
        if (byte >= 10) {
            if (written < (int)buf_size - 1) buf[written++] = '0' + ((byte / 10) % 10);
        }
        if (written < (int)buf_size - 1) buf[written++] = '0' + (byte % 10);
        if (i > 0 && written < (int)buf_size - 1) buf[written++] = '.';
    }
    buf[written] = 0;
}

bool triad_firewall_match_host_pattern(const char *hostname, const char *pattern) {
    if (!hostname || !pattern) return false;

    if (pattern[0] == '*') {
        const char *dot = hostname;
        while (*dot && *dot != '.') dot++;
        if (*dot == '.') {
            return triad_firewall_match_host_pattern(dot + 1, pattern + 2);
        }
        return false;
    }

    while (*hostname && *pattern) {
        if (*pattern == '*') {
            return true;
        }
        if (*hostname != *pattern) {
            return false;
        }
        hostname++;
        pattern++;
    }

    return *hostname == 0 && *pattern == 0;
}

int triad_firewall_load_default_rules(void) {
    FirewallRule rule;

    rule.src_ip = 0;
    rule.src_mask = 0xFFFFFFFF;
    rule.src_port_start = 0;
    rule.src_port_end = 65535;
    rule.dst_ip = 0;
    rule.dst_mask = 0;
    rule.dst_port_start = 22;
    rule.dst_port_end = 22;
    rule.protocol = FW_PROTO_TCP;
    rule.proc_tier = PROC_T0_KERNEL;
    rule.action = FW_ALLOW;
    rule.log = 0;
    rule.enabled = 1;
    rule.description[0] = 'S';
    rule.description[1] = 'S';
    rule.description[2] = 'H';
    rule.description[3] = 0;
    triad_firewall_add_rule(&rule);

    rule.dst_port_start = 80;
    rule.dst_port_end = 80;
    rule.proc_tier = PROC_T1_SYSTEM;
    rule.description[0] = 'H';
    rule.description[1] = 'T';
    rule.description[2] = 'T';
    rule.description[3] = 'P';
    rule.description[4] = 0;
    triad_firewall_add_rule(&rule);

    rule.dst_port_start = 443;
    rule.dst_port_end = 443;
    rule.description[0] = 'H';
    rule.description[1] = 'T';
    rule.description[2] = 'T';
    rule.description[3] = 'P';
    rule.description[4] = 'S';
    rule.description[5] = 0;
    triad_firewall_add_rule(&rule);

    rule.src_ip = triad_firewall_parse_ip("10.0.0.0");
    rule.src_mask = triad_firewall_parse_ip("255.0.0.0");
    rule.dst_port_start = 0;
    rule.dst_port_end = 0;
    rule.protocol = FW_PROTO_ANY;
    rule.proc_tier = PROC_T0_KERNEL;
    rule.action = FW_DENY;
    rule.description[0] = 'B';
    rule.description[1] = 'l';
    rule.description[2] = 'o';
    rule.description[3] = 'c';
    rule.description[4] = 'k';
    rule.description[5] = ' ';
    rule.description[6] = '1';
    rule.description[7] = '0';
    rule.description[8] = '.';
    rule.description[9] = '0';
    rule.description[10] = 0;
    triad_firewall_add_rule(&rule);

    triad_firewall_add_host_pattern("*.internal.local", FW_DENY, PROC_T3_USER);
    triad_firewall_add_host_pattern("*.admin.*", FW_DENY, PROC_T4_AI);

    return 0;
}