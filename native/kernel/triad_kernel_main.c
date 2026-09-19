#include "triad_kernel.h"
#include "triad_mm.h"
#include "triad_vfs.h"
#include "triad_sched.h"
#include "triad_syscall.h"
#include "triad_ai.h"
#include "triad_equilibrium.h"
#include "triad_sandbox.h"
#include "triad_gdt_idt.h"
#include "triad_isr.h"
#include "triad_pic.h"
#include "triad_pit.h"
#include "triad_paging.h"
#include "triad_serial.h"
#include "triad_runtime_svc.h"
#include "triad_tri_loader.h"
#include "triad_interpreter.h"
#include "triad_shell.h"
#include "triad_keyboard.h"
#include "triad_user.h"
#include "triad_ata.h"
#include "triad_disk.h"
#include "triad_net.h"
#include "triad_rtl8139.h"
#include "triad_stack_protect.h"
#include "triad_aslr.h"
#include "triad_crypto.h"
#include "triad_audit.h"
#include "triad_firewall.h"
#include "triad_hardening.h"
#include "triad_signal.h"
#include <string.h>

TriadProc triad_procs[MAX_PROCS];
int triad_current_pid = 0;

static void print_banner(void) {
    triad_serial_puts("\n\n");
    triad_serial_puts("========================================\n");
    triad_serial_puts("  TriadLang OS v2.0 — bare metal kernel\n");
    triad_serial_puts("  P1 (phase) + P2 (memory) + P3 (FDT)\n");
    triad_serial_puts("  AI: native, always learning\n");
    triad_serial_puts("  Memory: paging enabled\n");
    triad_serial_puts("  Scheduler: preemptive (PIT 100Hz)\n");
    triad_serial_puts("  VFS: in-memory + DiskFS\n");
    triad_serial_puts("  Disk: ATA/IDE driver\n");
    triad_serial_puts("  Network: TCP/IP + RTL8139\n");
    triad_serial_puts("  User: Ring 3 with TSS\n");
    triad_serial_puts("  Security: NX+SMEP+SMAP+KPTI+W^X+ASLR+Canary\n");
    triad_serial_puts("========================================\n\n");
}

static void proc_init(int pid, TriadTier tier, const char *name) {
    if (pid < 0 || pid >= MAX_PROCS) return;
    if (!name) name = "?";
    TriadProc *p = &triad_procs[pid];
    p->id = pid;
    p->tier = tier;
    p->alive = true;
    p->parent_id = -1;
    p->memory_used = 0;
    int k = 0;
    while (name[k] && k < 63) { p->name[k] = name[k]; k++; }
    p->name[k] = 0;
    p->cwd[0] = '/';
    p->cwd[1] = 0;
    p->caps.bits = 0;
    p->caps.bounding = 0xFFFFFFFF;
    p->caps.ambient = 0;
    for (int i = 0; i < MAX_FDS; i++) {
        p->fds[i].used = false;
    }
    if (pid == 0) {
        p->caps.bits = 0xFFFFFFFF;
        p->caps.bounding = 0xFFFFFFFF;
    } else if (tier == PROC_T1_SYSTEM) {
        p->caps.bits = CAP_FS_READ | CAP_FS_WRITE | CAP_NET_GET | CAP_AI_TOOL;
        p->caps.bounding = CAP_FS_READ | CAP_FS_WRITE | CAP_FS_REMOVE | CAP_NET_GET | CAP_NET_POST | CAP_AI_TOOL | CAP_ML_LOCAL;
    } else if (tier == PROC_T3_USER) {
        p->caps.bits = CAP_FS_READ | CAP_FS_WRITE;
        p->caps.bounding = CAP_FS_READ | CAP_FS_WRITE;
    } else if (tier == PROC_T4_AI) {
        p->caps.bits = CAP_AI_TOOL | CAP_ML_LOCAL;
        p->caps.bounding = CAP_AI_TOOL | CAP_ML_LOCAL;
    }
    p->sandbox = triad_sb_new("/");
}

static void test_thread_a(void *arg) {
    (void)arg;
    int count = 0;
    while (count < 5) {
        triad_serial_puts("[thread-A] running ");
        triad_serial_dec(count);
        triad_serial_puts("\n");
        for (volatile int i = 0; i < 1000000; i++) { }
        count++;
    }
}

static void test_thread_b(void *arg) {
    (void)arg;
    int count = 0;
    while (count < 5) {
        triad_serial_puts("[thread-B] running ");
        triad_serial_dec(count);
        triad_serial_puts("\n");
        for (volatile int i = 0; i < 1000000; i++) { }
        count++;
    }
}

static void kernel_test_disk(void) {
    triad_serial_puts("[kernel] testing disk driver...\n");
    
    AtaDrive drive0, drive1;
    if (triad_ata_identify(0, 0, &drive0) == 0) {
        triad_serial_puts("[disk] primary master: ");
        triad_serial_puts(drive0.model);
        triad_serial_puts(" (");
        triad_serial_dec(drive0.sectors / 2 / 1024);
        triad_serial_puts(" MB)\n");
    }
    if (triad_ata_identify(0, 1, &drive1) == 0) {
        triad_serial_puts("[disk] primary slave: ");
        triad_serial_puts(drive1.model);
        triad_serial_puts("\n");
    }
}

static void kernel_test_net(void) {
    triad_serial_puts("[kernel] testing network stack...\n");
    
    NetInterface iface;
    iface.id = 0;
    iface.name[0] = 'e';
    iface.name[1] = 't';
    iface.name[2] = 'h';
    iface.name[3] = '0';
    iface.name[4] = 0;
    iface.mac.bytes[0] = 0x52;
    iface.mac.bytes[1] = 0x54;
    iface.mac.bytes[2] = 0x00;
    iface.mac.bytes[3] = 0x12;
    iface.mac.bytes[4] = 0x34;
    iface.mac.bytes[5] = 0x56;
    iface.state = NET_IF_UP;
    
    int if_id = triad_net_register_interface(&iface);
    triad_serial_puts("[net] registered interface eth0, id=");
    triad_serial_dec(if_id);
    triad_serial_puts("\n");
    
    char ipbuf[16];
    net_ip4_to_str(iface.ip, ipbuf);
    triad_serial_puts("[net] ip: ");
    triad_serial_puts(ipbuf);
    triad_serial_puts("\n");
}

static void kernel_test_full(void) {
    triad_serial_puts("[kernel] testing syscall layer...\n");
    triad_syscall_test();
    triad_serial_puts("[kernel] ai observations: ");
    triad_serial_dec(triad_ai_observation_count());
    triad_serial_puts("\n");

    triad_serial_puts("[kernel] creating /etc/init.tri...\n");
    triad_vfs_create("/etc/init.tri");
    int fd = triad_vfs_open("/etc/init.tri", 1);
    const char init_src[] = "print(\"hello from triadlang os\");";
    triad_vfs_write(fd, init_src, sizeof(init_src) - 1);
    triad_vfs_close(fd);

    TriadVfsNode stat;
    if (triad_vfs_stat("/etc/init.tri", &stat) == 0) {
        triad_serial_puts("[kernel] /etc/init.tri size=");
        triad_serial_dec(stat.size);
        triad_serial_puts("\n");
    }

    triad_serial_puts("[kernel] testing .tri loader...\n");
    TriModuleInfo mod_info;
    if (triad_tri_load_from_vfs("/etc/init.tri", &mod_info) == 0) {
        triad_serial_puts("[tri_loader] parsed: ");
        triad_serial_puts(mod_info.source_path);
        triad_serial_puts(" stmts=");
        triad_serial_dec(mod_info.n_statements);
        triad_serial_puts(" fns=");
        triad_serial_dec(mod_info.n_functions);
        triad_serial_puts(" imports=");
        triad_serial_dec(mod_info.n_imports);
        triad_serial_puts("\n");
    }

    triad_serial_puts("[kernel] loading program via runtime service...\n");
    triad_rt_svc_init();
    int prog_id = triad_rt_svc_load("/etc/init.tri", 0);
    if (prog_id >= 0) {
        triad_serial_puts("[rt_svc] program loaded, id=");
        triad_serial_dec(prog_id);
        triad_serial_puts("\n");
        triad_rt_svc_run(prog_id);
        RtProgram status;
        if (triad_rt_svc_status(prog_id, &status) == 0) {
            triad_serial_puts("[rt_svc] program completed\n");
        }
    }

    triad_serial_puts("[kernel] equilibrium states:\n");
    const char *domain_names[] = {"CPU", "MEM", "NET", "GPU", "DISK", "THERM"};
    for (int i = 0; i < 6; i++) {
        EqHwDomain d = (EqHwDomain)i;
        double u;
        triad_eq_read(d, &u);
        triad_serial_puts("  ");
        triad_serial_puts(domain_names[i]);
        triad_serial_puts(": util=");
        triad_serial_float(u);
        triad_serial_puts("\n");
    }

    triad_serial_puts("[kernel] paging: free=");
    triad_serial_dec(triad_paging_free_pages());
    triad_serial_puts(" used=");
    triad_serial_dec(triad_paging_used_pages());
    triad_serial_puts("\n");

    triad_serial_puts("[kernel] ai recall test:\n");
    char ai_buf[512];
    int n = triad_ai_query("hello", ai_buf, sizeof(ai_buf));
    if (n > 0) {
        triad_serial_puts("  recall: ");
        triad_serial_puts(ai_buf);
        triad_serial_puts("\n");
    }

    kernel_test_disk();
    kernel_test_net();

    triad_serial_puts("[kernel] scheduler: preemptive enabled\n");
    
    triad_serial_puts("[kernel] creating test threads...\n");
    TriadThread thread_a, thread_b;
    int ta = triad_thread_create(&thread_a, test_thread_a, NULL);
    int tb = triad_thread_create(&thread_b, test_thread_b, NULL);
    triad_serial_puts("[kernel] thread A id=");
    triad_serial_dec(ta);
    triad_serial_puts(" thread B id=");
    triad_serial_dec(tb);
    triad_serial_puts("\n");
    triad_serial_puts("[kernel] threads will run concurrently via preemptive scheduling\n");
}

void kernel_main(void) {
    triad_serial_init();
    triad_serial_puts("[boot] initializing gdt...\n");
    gdt_init();

    triad_serial_puts("[boot] initializing pic...\n");
    triad_pic_init();

    triad_serial_puts("[boot] initializing idt + isr...\n");
    isr_table_init();
    idt_init();

    triad_serial_puts("[boot] initializing memory manager...\n");
    triad_mm_init(HEAP_SIZE);

    triad_serial_puts("[boot] initializing paging...\n");
    triad_paging_init(HEAP_SIZE);
    triad_mm_enable_guard_pages(1);

    triad_serial_puts("[boot] initializing stack canary...\n");
    triad_stack_canary_init();

    triad_serial_puts("[boot] initializing aslr...\n");
    triad_aslr_init();

    triad_serial_puts("[boot] initializing crypto...\n");
    uint8_t rng_test[16];
    triad_random_bytes(rng_test, 16);

    triad_serial_puts("[boot] initializing audit...\n");
    triad_audit_init();

    triad_serial_puts("[boot] initializing firewall...\n");
    triad_firewall_init();

    triad_serial_puts("[boot] initializing hardening...\n");
    triad_hardening_init();

    triad_serial_puts("[boot] initializing KPTI...\n");
    if (triad_kpti_init() == 0) {
        triad_enable_kpti();
        triad_serial_puts("[boot] KPTI enabled (kernel page table isolation)\n");
    } else {
        triad_serial_puts("[boot] KPTI initialization failed\n");
    }

    triad_hardening_report();

    triad_serial_puts("[boot] initializing signals...\n");
    triad_signal_init();

    triad_serial_puts("[boot] initializing ata driver...\n");
    triad_ata_init();

    triad_serial_puts("[boot] initializing disk filesystem...\n");
    triad_disk_init();

    triad_serial_puts("[boot] initializing network stack...\n");
    triad_net_init();

    triad_serial_puts("[boot] initializing vfs...\n");
    triad_vfs_init();
    triad_vfs_mkdir("/etc");
    triad_vfs_mkdir("/home");
    triad_vfs_mkdir("/tmp");
    triad_vfs_mkdir("/proc");
    triad_vfs_mkdir("/dev");
    triad_vfs_mkdir("/sys");
    triad_vfs_mkdir("/models");
    triad_vfs_mkdir("/var");
    triad_vfs_mkdir("/var/log");

    triad_serial_puts("[boot] creating /dev/random and /dev/urandom...\n");
    triad_vfs_create("/dev/random");
    triad_vfs_create("/dev/urandom");

    triad_serial_puts("[boot] initializing scheduler...\n");
    triad_sched_init();

    triad_serial_puts("[boot] initializing ai engine...\n");
    triad_ai_init();

    triad_serial_puts("[boot] initializing equilibrium...\n");
    triad_eq_init();

    triad_serial_puts("[boot] initializing user mode...\n");
    triad_user_init();

    triad_serial_puts("[boot] initializing syscalls...\n");
    triad_syscall_init();

    triad_serial_puts("[boot] initializing runtime service...\n");
    triad_rt_svc_init();

    triad_serial_puts("[boot] initializing keyboard...\n");
    triad_keyboard_init();

    triad_serial_puts("[boot] initializing shell...\n");
    triad_shell_init();

    triad_serial_puts("[boot] creating kernel process...\n");
    for (int i = 0; i < MAX_PROCS; i++) {
        triad_procs[i].alive = false;
    }
    proc_init(0, PROC_T0_KERNEL, "kernel");
    triad_current_pid = 0;

    triad_audit_log(AUDIT_PROC_CREATE, AUDIT_SEV_INFO, "kernel process created pid=0");

    print_banner();

    triad_serial_puts("[boot] security features enabled:\n");
    if (triad_has_nx()) triad_serial_puts("  - NX (No-Execute)\n");
    if (triad_has_smep()) triad_serial_puts("  - SMEP (Supervisor Mode Execution Prevention)\n");
    if (triad_has_smap()) triad_serial_puts("  - SMAP (Supervisor Mode Access Prevention)\n");
    if (triad_has_kpti()) triad_serial_puts("  - KPTI (Kernel Page Table Isolation)\n");
    if (triad_has_wx()) triad_serial_puts("  - W^X Enforcement (text=RX, rodata=R, bss=RW)\n");
    if (triad_has_rdrand()) triad_serial_puts("  - RDRAND (Hardware RNG)\n");
    if (triad_has_rdseed()) triad_serial_puts("  - RDSEED (Hardware Entropy)\n");
    triad_serial_puts("  - Stack canary initialized\n");
    triad_serial_puts("  - ASLR enabled\n");
    triad_serial_puts("  - Heap guard pages enabled\n");
    triad_serial_puts("  - Audit logging active\n");
    triad_serial_puts("  - Firewall default-deny\n");
    triad_serial_puts("  - Signal delivery system active\n");
    triad_serial_puts("  - /dev/random + /dev/urandom available\n\n");

    triad_audit_log(AUDIT_PROC_CREATE, AUDIT_SEV_INFO, "boot sequence initiated");

    triad_serial_puts("[boot] observing boot into ai...\n");
    triad_ai_observe(0, "kernel", "boot complete: paging+gdt+idt+pic+pit+ata+disk+net+vfs+sched+ai+eq+user+syscalls+rt_svc+tri_loader+interp+shell+security+kpti+wx+signals");

    triad_serial_puts("[boot] enabling timer (PIT 100Hz)...\n");
    triad_pit_init(TIMER_HZ);
    triad_pic_unmask(0);
    irq_enable();

    triad_serial_puts("[boot] system ready.\n\n");

    kernel_test_full();

    triad_serial_puts("\n[kernel] launching interactive shell...\n");
    triad_shell_run();

    irq_disable();
    while (1) {
        __asm__ volatile ("hlt");
    }
}
