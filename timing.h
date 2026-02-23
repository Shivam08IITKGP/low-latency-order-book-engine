#pragma once
#include <cstdint>
#include <chrono>
#include <xmmintrin.h>

// --------------------------------------------------------------------
// RDTSC/RDTSCP - Intel's Recommended Timing Sandwich
// --------------------------------------------------------------------
// Per-message measurement strategy:
//
//   START: LFENCE + RDTSC  — LFENCE drains in-flight loads (~20 cycles),
//          ensuring prior instructions complete before we read the counter.
//          We use LFENCE here instead of CPUID because CPUID (~100-150
//          cycles) fully serializes the pipeline and becomes the dominant
//          cost, inflating every latency sample and throttling throughput.
//
//   END:   RDTSCP + LFENCE — RDTSCP waits for prior instructions to retire
//          before reading TSC, then the trailing LFENCE prevents subsequent
//          instructions from executing before the read is complete.
//
// Total overhead: ~40-50 cycles per sandwich vs ~250+ with CPUID on both sides.
//
// rdtsc_start_full() with CPUID is kept for the outer total-run measurement
// only (called once, not per-message) where accuracy beats throughput.

// Full serialization (CPUID) — use ONLY for outer benchmark boundaries
inline uint64_t rdtsc_start_full()
{
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    __asm__ __volatile__ (
        "cpuid\n\t"
        "rdtsc\n\t"
        : "=a" (lo), "=d" (hi)
        :: "rbx", "rcx");
    return ((uint64_t)hi << 32) | lo;
#else
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
}

// Lightweight serialization (LFENCE) — use for per-message start timestamps
inline uint64_t rdtsc_start()
{
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    __asm__ __volatile__ (
        "lfence\n\t"
        "rdtsc\n\t"
        : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
}

inline uint64_t rdtsc_end()
{
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi, aux;
    __asm__ __volatile__ (
        "rdtscp\n\t"
        "lfence\n\t"
        : "=a" (lo), "=d" (hi), "=c" (aux));
    return ((uint64_t)hi << 32) | lo;
#else
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
}

// Fast RDTSC for timestamps in hot path (NO serialization overhead)
// Use for event timestamps, NOT for measurement boundaries
inline uint64_t rdtsc_fast()
{
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
}

// Global CPU frequency (cycles per nanosecond), initialized at startup
extern double g_cycles_per_ns;

// Calibrate CPU frequency by measuring cycles over a known wall-clock period
double calibrate_cpu_frequency();

// Convert TSC cycles to nanoseconds
inline uint64_t cycles_to_ns(uint64_t cycles)
{
    return static_cast<uint64_t>(cycles / g_cycles_per_ns);
}

// Get current timestamp in nanoseconds (hot-path safe, no CPUID)
inline uint64_t get_timestamp_ns()
{
    return cycles_to_ns(rdtsc_fast());
}