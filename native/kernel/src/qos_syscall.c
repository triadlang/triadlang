#include "qos_integration.h"
#include "triad_kernel.h"
#include "triad_vfs.h"

// Extensão dos syscalls tradicionais para incluir operações quânticas
// O usuário continua usando syscalls normais, mas por baixo funciona de forma diferente

static int64_t syscall_dispatch_extended(uint64_t num, uint64_t a, uint64_t b, 
                                          uint64_t c, uint64_t d, uint64_t e) {
    switch (num) {
        // Syscalls tradicionais (pass-through)
        case SYS_EXIT:       // 0
            return 0;
            
        case SYS_WRITE:      // 1
            return triad_vfs_write((int)a, (const void *)b, (uint32_t)c);
            
        case SYS_READ:       // 2
            return triad_vfs_read((int)a, (void *)b, (uint32_t)c);
            
        case SYS_OPEN:       // 3
            return triad_vfs_open((const char *)a, (uint8_t)b);
            
        case SYS_CLOSE:      // 4
            return triad_vfs_close((int)a);
            
        case SYS_SEEK:       // 5
            return triad_vfs_seek((int)a, (uint32_t)b);
            
        case SYS_STAT:       // 6
            return triad_vfs_stat((const char *)a, (TriadVfsNode *)b);
            
        case SYS_MKDIR:      // 7
            return triad_vfs_mkdir((const char *)a);
            
        case SYS_LISTDIR:    // 8
            return triad_vfs_listdir((const char *)a, (char (*)[256])b, (int)c);
            
        case SYS_REMOVE:     // 9
            return triad_vfs_remove((const char *)a);
            
        case SYS_SPAWN:      // 10
            return qos_proc_from_tri_file((const char *)a);
            
        case SYS_WAIT:       // 11
            // Esperar processo cristalizar
            if (a >= 0 && a < MAX_PROCS) {
                qos_wait_equilibrium((int)a, 0.01);
                return 0;
            }
            return -1;
            
        case SYS_GETPID:     // 12
            return qos_sched_get_current();
            
        case SYS_GETTICKS:   // 13
            // Retorna tempo global quântico
            return (int64_t)(g_qos_sched.global_time * 1000);
            
        case SYS_SLEEP:      // 14
            // Sleep = evoluir o campo
            qos_evolve_all((double)a / 1000.0);
            return 0;
            
        case SYS_YIELD:      // 25
            qos_sched_yield();
            return 0;
            
        // Syscalls quânticos (extensão)
        case SYS_QUANTUM_CREATE:  // 50
            return sys_quantum_create(a, b, c);
            
        case SYS_QUANTUM_EVOLVE:  // 51
            return sys_quantum_evolve(a, b);
            
        case SYS_QUANTUM_MEASURE: // 52
            return sys_quantum_measure(a, b);
            
        case SYS_QUANTUM_COUPLE:  // 53
            return sys_quantum_couple(a, b, c);
            
        case SYS_QUANTUM_DECOUPLE: // 54
            qos_decouple((int)a, (int)b);
            return 0;
            
        case SYS_QUANTUM_SAVE:    // 55
            return qos_vfs_save_quantum_state((const char *)a, (int)b);
            
        case SYS_QUANTUM_LOAD:    // 56
            return qos_vfs_load_quantum_state((const char *)a, (int)b);
            
        case SYS_LLM_GENERATE:    // 60
            return sys_llm_generate(a, b, c);
            
        case SYS_LLM_ABSORB:      // 61
            return sys_llm_absorb(a, b);
            
        case SYS_LLM_SUGGEST:     // 62
            return qos_llm_suggest((const char *)a, (char *)b);
            
        // Syscalls de IA/ML (extensão)
        case SYS_AI_QUERY:        // 17
            return -1;  // TODO
            
        case SYS_AI_RECORD:       // 18
            qos_llm_absorb_text((const char *)a);
            return 0;
            
        case SYS_SOLVER:          // 19
            // Evoluir solver quântico
            qos_evolve_all((double)a);
            return 0;
            
        case SYS_EQUILIBRIUM:     // 24
            // Verificar equilíbrio
            return g_qos_sched.quantum_procs[a].crystallized;
            
        default:
            return -1;
    }
}

// Enum para syscalls quânticos
enum {
    SYS_QUANTUM_CREATE = 50,
    SYS_QUANTUM_EVOLVE = 51,
    SYS_QUANTUM_MEASURE = 52,
    SYS_QUANTUM_COUPLE = 53,
    SYS_QUANTUM_DECOUPLE = 54,
    SYS_QUANTUM_SAVE = 55,
    SYS_QUANTUM_LOAD = 56,
    
    SYS_LLM_GENERATE = 60,
    SYS_LLM_ABSORB = 61,
    SYS_LLM_SUGGEST = 62,
};