#include "timing.h"
#include <iostream>
#include <iomanip>

double g_cycles_per_ns = 0.0;

double calibrate_cpu_frequency()
{
#if defined(__x86_64__) || defined(_M_X64)
    // Full CPUID serialization here is correct — calibration runs once at startup
    uint64_t start_tsc = rdtsc_start_full();
    auto start_time    = std::chrono::high_resolution_clock::now();

    for (volatile int i = 0; i < 50000000; i++);

    uint64_t end_tsc  = rdtsc_end();
    auto end_time     = std::chrono::high_resolution_clock::now();

    uint64_t cycles    = end_tsc - start_tsc;
    auto duration_ns   = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         end_time - start_time).count();

    return static_cast<double>(cycles) / duration_ns;
#else
    return 1.0;
#endif
}