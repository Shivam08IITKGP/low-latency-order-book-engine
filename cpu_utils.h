#pragma once
#include <cstdint>

// --------------------------------------------------------------------
// CPU ISOLATION AND PINNING
// --------------------------------------------------------------------
// CRITICAL: Thread affinity (pthread_setaffinity_np) is NOT enough!
//
// The kernel can still interrupt a pinned core for:
//   - Network interrupts
//   - Timer ticks
//   - RCU callbacks
//   - System tasks
//
// PRODUCTION REQUIREMENT: Boot-level CPU isolation.
// Add to GRUB config (/etc/default/grub):
//   GRUB_CMDLINE_LINUX="isolcpus=2 nohz_full=2 rcu_nocbs=2"
//
//   isolcpus=2  : Removes core from scheduler load balancing
//   nohz_full=2 : Disables periodic timer ticks (adaptive-tick mode)
//   rcu_nocbs=2 : Offloads RCU callbacks to other cores
//
// Then: sudo update-grub && sudo reboot
// Expected jitter reduction: 400μs → <5μs

bool isCoreIsolated(int core_id);
bool pinThreadToCore(int core_id);

// --------------------------------------------------------------------
// HARDWARE PAUSE - Busy-wait without yielding the core
// --------------------------------------------------------------------
// HFT Design: Never yield() the core! Context switches cost 1–3μs.
// CPU PAUSE hints the processor we're spinning without giving up the core.
inline void cpu_pause()
{
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();          // x86 PAUSE (~10 cycles)
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    asm volatile("" ::: "memory");   // Compiler barrier fallback
#endif
}