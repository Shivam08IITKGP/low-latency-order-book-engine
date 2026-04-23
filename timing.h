#pragma once
#include <cstdint>
#include <chrono>
#include <xmmintrin.h>

/**
 * RDTSC/RDTSCP SERALIZATION STRATEGY
 * 
 * To achieve nanosecond-level precision, we must prevent the CPU from reordering
 * instructions around the timestamp read.
 * 
 * Measurement Window:
 * [LFENCE] -> Drain in-flight loads
 * [RDTSC]  -> Sample timer
 * ... <Critical Path> ...
 * [RDTSCP] -> Wait for instructions to retire + Sample timer
 * [LFENCE] -> Prevent subsequent instructions from leaking into the window
 * 
 * Overhead: ~40-50 cycles per sample.
 */

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

extern double g_cycles_per_ns;

double calibrate_cpu_frequency();

inline uint64_t cycles_to_ns(uint64_t cycles)
{
    return static_cast<uint64_t>(cycles / g_cycles_per_ns);
}

inline uint64_t get_timestamp_raw()
{
    return rdtsc_fast();
}