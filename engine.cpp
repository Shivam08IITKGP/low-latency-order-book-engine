#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <map>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <cstring>
#include <sched.h> // For CPU pinning
#include <pthread.h>
#include <iomanip>
#include <xmmintrin.h> // For _mm_lfence() memory fence intrinsic
#include <array>
#include "common.h"

/*
 * Low-Latency Order Book Implementation
 *
 * DESIGN DECISIONS:
 * - Price levels stored in flat vectors indexed by price.
 * - Orders stored in a pre-allocated vector indexed by order ID.
 *
 * Benefits:
 * 1. O(1) order lookup (direct indexing).
 * 2. No hash collisions (vs unordered_map).
 * 3. No rehashing.
 * 4. Excellent cache locality.
 * 5. No heap allocations in hot path.
 */

const uint64_t MAX_PRICE = 100000;

// --------------------------------------------------------------------
// RING BUFFER for publishing updates to separate thread
// --------------------------------------------------------------------
struct UpdateMessage
{
    // Reordered from largest to smallest
    uint64_t order_id;
    uint64_t price;
    uint64_t timestamp_ns;
    uint32_t quantity;
    char type; // 'N'=New, 'X'=Cancel, 'T'=Trade, 'M'=Modify
    char side;
};

template<typename T, size_t Size>
class RingBuffer
{
    // Ensure Size is a power of 2 for bitwise optimization
    static_assert((Size != 0) && ((Size & (Size - 1)) == 0), "Size must be a power of 2");

private:
    alignas(64) T buffer[Size];
    
    // Producer-exclusive variables (Core 2)
    alignas(64) std::atomic<size_t> write_idx{0};
    size_t cached_read_idx{0}; // Lives in Core 2's L1 cache
    
    // Consumer-exclusive variables (Core 3)
    alignas(64) std::atomic<size_t> read_idx{0};
    size_t cached_write_idx{0}; // Lives in Core 3's L1 cache
    
    // Bitmask for fast modulo (e.g., 1048576 - 1 = 1048575)
    static constexpr size_t MASK = Size - 1;

public:
    bool push(const T& msg)
    {
        size_t current_write = write_idx.load(std::memory_order_relaxed);
        // Bitwise AND is significantly faster than modulo (%)
        size_t next_write = (current_write + 1) & MASK; 
        
        // 1. Fast path: Check against our private, cached copy of read_idx
        if (next_write == cached_read_idx)
        {
            // 2. Slow path: We MIGHT be full. Go fetch the actual atomic read_idx from Core 3
            cached_read_idx = read_idx.load(std::memory_order_acquire);
            
            // 3. Are we actually full?
            if (next_write == cached_read_idx)
                return false; 
        }
        
        buffer[current_write] = msg;
        write_idx.store(next_write, std::memory_order_release);
        return true;
    }
    
    bool pop(T& msg)
    {
        size_t current_read = read_idx.load(std::memory_order_relaxed);
        
        // 1. Fast path: Check against our private, cached copy of write_idx
        if (current_read == cached_write_idx)
        {
            // 2. Slow path: We MIGHT be empty. Go fetch the actual atomic write_idx from Core 2
            cached_write_idx = write_idx.load(std::memory_order_acquire);
            
            // 3. Are we actually empty?
            if (current_read == cached_write_idx)
                return false; 
        }
        
        msg = buffer[current_read];
        read_idx.store((current_read + 1) & MASK, std::memory_order_release);
        return true;
    }
    
    bool isEmpty() const
    {
        return read_idx.load(std::memory_order_acquire) == 
               write_idx.load(std::memory_order_acquire);
    }

    void pre_fault_memory() 
    {
        // Touch one element per 4KB page to force OS to allocate physical memory
        // This eliminates page faults during hot path execution
        const size_t page_size = 4096;
        // Stride = elements per page (no +1! That skips pages)
        for (size_t i = 0; i < Size; i += page_size / sizeof(T)) 
        {
            buffer[i] = T(); 
        }
    }
};

// Global ring buffer - 1M slots should handle high throughput
RingBuffer<UpdateMessage, 1048576> updateBuffer;
std::atomic<bool> stopPublisher{false};

// --------------------------------------------------------------------
// ZERO-COPY PACKET VIEW for lock-free communication between threads
// --------------------------------------------------------------------
// In production HFT: NIC streams packets directly into user-space ring buffer via DMA
// Engine reads pointer to packet, never copies payload
// Benefits: 16 bytes (pointer view) vs 136 bytes (full copy)
// Result: 4 messages fit in single 64-byte cache line, 8.5x less interconnect traffic
struct PacketView
{
    const char* payload; // Direct pointer to mmap file / DMA buffer (zero-copy!)
    char msg_type;       // Cached type for immediate switch() branching
};

// Input queue: Network thread produces, Engine thread consumes
RingBuffer<PacketView, 524288> inputQueue;
std::atomic<bool> stopNetworkThread{false};
std::atomic<bool> stopEngine{false};  // Signal engine to stop after network is done

// --------------------------------------------------------------------
// CPU ISOLATION AND PINNING
// --------------------------------------------------------------------
// CRITICAL: Thread affinity (pthread_setaffinity_np) is NOT enough!
//
// pthread_setaffinity_np() tells the OS "run this thread on Core 2"
// BUT the kernel can still interrupt Core 2 for:
//   - Network interrupts
//   - Timer ticks
//   - RCU callbacks
//   - System tasks
//
// PRODUCTION REQUIREMENT: Boot-level CPU isolation
// Add to GRUB configuration (/etc/default/grub):
//   GRUB_CMDLINE_LINUX="isolcpus=2 nohz_full=2 rcu_nocbs=2"
//
// What each parameter does:
//   isolcpus=2      : Removes Core 2 from scheduler's load balancing
//   nohz_full=2     : Disables timer ticks on Core 2 (adaptive-tick mode)
//   rcu_nocbs=2     : Offloads RCU callbacks from Core 2 to other cores
//
// Then run: sudo update-grub && sudo reboot
//
// Expected jitter reduction: 400μs → <5μs

// Check if a core is isolated from the kernel scheduler
bool isCoreIsolated(int core_id)
{
    std::ifstream isolcpus("/sys/devices/system/cpu/isolated");
    if (!isolcpus.is_open())
        return false;
    
    std::string isolated;
    std::getline(isolcpus, isolated);
    
    // Parse isolated cores (format: "2" or "2-3" or "2,4")
    return isolated.find(std::to_string(core_id)) != std::string::npos;
}

bool pinThreadToCore(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    
    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (result != 0)
    {
        std::cerr << "[ERROR] Failed to pin thread to core " << core_id << "\n";
        return false;
    }
    
    std::cout << "[CPU] Thread pinned to core " << core_id;
    
    // Check if core is properly isolated for HFT workloads
    if (!isCoreIsolated(core_id))
    {
        std::cout << " [WARNING: NOT ISOLATED]";
    }
    
    std::cout << "\n";
    return true;
}

// --------------------------------------------------------------------
// HARDWARE PAUSE - Busy-wait without yielding core
// --------------------------------------------------------------------
// HFT Design: Never yield() the core! Context switches cost 1-3μs.
// Instead, use CPU PAUSE instruction to hint we're spinning.
// This prevents pipeline speculation without giving up the core.
inline void cpu_pause()
{
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();  // x86 PAUSE instruction (~10 cycles)
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");  // ARM yield hint
#else
    // Fallback for other architectures (still better than yield)
    asm volatile("" ::: "memory");
#endif
}

// --------------------------------------------------------------------
// RDTSC/RDTSCP - Intel's Recommended Timing Sandwich
// --------------------------------------------------------------------
// Industry standard pattern for accurate cycle counting:
//   1. CPUID/LFENCE - Serialize pipeline (all prior instructions complete)
//   2. RDTSC - Read start timestamp
//   3. [Code to measure]
//   4. RDTSCP - Read end timestamp (waits for measured code to finish)
//   5. LFENCE - Ensure RDTSCP completes before subsequent code
//
// CPUID is fully serializing (better than LFENCE alone)
// RDTSCP is partially serializing (waits for prior ops, not subsequent)
inline uint64_t rdtsc_start()
{
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    // CPUID serializes: clears pipeline, waits for all prior instructions
    __asm__ __volatile__ (
        "cpuid\n\t"
        "rdtsc\n\t"
        : "=a" (lo), "=d" (hi)
        :: "rbx", "rcx");  // CPUID clobbers rbx, rcx
    return ((uint64_t)hi << 32) | lo;
#else
    return std::chrono::high_resolution_clock::now()
           .time_since_epoch().count();
#endif
}

inline uint64_t rdtsc_end()
{
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi, aux;
    // RDTSCP: Waits for prior instructions, then reads TSC
    // LFENCE: Prevents subsequent instructions from starting early
    __asm__ __volatile__ (
        "rdtscp\n\t"
        "lfence\n\t"
        : "=a" (lo), "=d" (hi), "=c" (aux));
    return ((uint64_t)hi << 32) | lo;
#else
    return std::chrono::high_resolution_clock::now()
           .time_since_epoch().count();
#endif
}

// Fast RDTSC for timestamps in hot path (NO serialization)
// Use this for event timestamps, NOT for measurement boundaries
inline uint64_t rdtsc_fast()
{
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return std::chrono::high_resolution_clock::now()
           .time_since_epoch().count();
#endif
}

// Calibrate CPU frequency for cycle -> nanosecond conversion
inline double calibrate_cpu_frequency()
{
#if defined(__x86_64__) || defined(_M_X64)
    // Measure cycles over a known time period
    uint64_t start_tsc = rdtsc_start();
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Busy-wait for ~100ms to get accurate measurement
    for (volatile int i = 0; i < 50000000; i++);
    
    uint64_t end_tsc = rdtsc_end();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    uint64_t cycles = end_tsc - start_tsc;
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       end_time - start_time).count();
    
    // cycles_per_ns = cycles / nanoseconds
    double cycles_per_ns = static_cast<double>(cycles) / duration_ns;
    return cycles_per_ns;
#else
    return 1.0;  // Fallback: already in nanoseconds
#endif
}

// Global CPU frequency (initialized at startup)
double g_cycles_per_ns = 0.0;

// Convert TSC cycles to nanoseconds
inline uint64_t cycles_to_ns(uint64_t cycles)
{
    return static_cast<uint64_t>(cycles / g_cycles_per_ns);
}

// Get timestamp in nanoseconds (for hot path - no serialization!)
// Uses fast rdtsc without CPUID overhead
inline uint64_t get_timestamp_ns()
{
    return cycles_to_ns(rdtsc_fast());
}

// --------------------------------------------------------------------
// LATENCY HISTOGRAM AND JITTER ANALYSIS
// --------------------------------------------------------------------
class LatencyHistogram
{
private:
    std::vector<uint64_t> buckets;
    const uint64_t bucket_size_ns;
    uint64_t max_latency = 0;
    uint64_t min_latency = UINT64_MAX;
    uint64_t total_samples = 0;
    
public:
    LatencyHistogram(uint64_t bucket_size = 10) : bucket_size_ns(bucket_size)
    {
        // Buckets: 0-10ns, 10-20ns, ..., 990-1000ns, 1000+ns
        buckets.resize(101, 0);
    }
    
    void addSample(uint64_t latency_ns)
    {
        total_samples++;
        max_latency = std::max(max_latency, latency_ns);
        min_latency = std::min(min_latency, latency_ns);
        
        size_t bucket = std::min(latency_ns / bucket_size_ns, static_cast<uint64_t>(buckets.size() - 1));
        buckets[bucket]++;
    }
    
    void printHistogram() const
    {
        std::cout << "\n========== LATENCY HISTOGRAM ==========\n";
        std::cout << "Min: " << min_latency << " ns\n";
        std::cout << "Max: " << max_latency << " ns (JITTER: " 
                  << (max_latency - min_latency) << " ns)\n";
        std::cout << "Bucket Size: " << bucket_size_ns << " ns\n\n";
        
        for (size_t i = 0; i < buckets.size(); i++)
        {
            if (buckets[i] == 0)
                continue;
                
            uint64_t range_start = i * bucket_size_ns;
            uint64_t range_end = (i == buckets.size() - 1) ? UINT64_MAX : (i + 1) * bucket_size_ns;
            
            double percentage = (100.0 * buckets[i]) / total_samples;
            
            std::cout << std::setw(6) << range_start << "-" 
                      << std::setw(6) << (range_end == UINT64_MAX ? "∞" : std::to_string(range_end))
                      << " ns: " << std::setw(8) << buckets[i] 
                      << " (" << std::fixed << std::setprecision(2) 
                      << percentage << "%)";
            
            // Visual bar
            int bar_length = static_cast<int>(percentage / 2);
            std::cout << " ";
            for (int j = 0; j < bar_length; j++)
                std::cout << "█";
            std::cout << "\n";
        }
        
        std::cout << "\nTotal Samples: " << total_samples << "\n";
        std::cout << "=======================================\n";
    }
    
    void printJitterAnalysis() const
    {
        uint64_t jitter = max_latency - min_latency;
        std::cout << "\n========== JITTER ANALYSIS ==========\n";
        std::cout << "Jitter (Max - Min): " << jitter << " ns\n";
        
        if (jitter < 1000)
            std::cout << "✓ Excellent: Very low jitter\n";
        else if (jitter < 5000)
            std::cout << "✓ Good: Acceptable jitter\n";
        else if (jitter < 10000)
            std::cout << "⚠ Warning: High jitter detected\n";
        else
            std::cout << "✗ Critical: Very high jitter (possible cache miss/page fault)\n";
        
        std::cout << "=====================================\n";
    }
};

// Zero-overhead latency recorder: flat pre-allocated buffer written from hot path
class LatencyRecorder
{
private:
    struct alignas(16) Sample { uint64_t latency_ns; uint32_t index; char type; };
    static constexpr size_t MAX_SAMPLES = 200000;
    std::array<Sample, MAX_SAMPLES> samples{}; // BSS/stack pre-allocated
    size_t count = 0;

public:
    // Inline, zero-branch hot-path write
    inline void addSample(uint64_t latency_ns, char type, uint32_t index)
    {
        samples[count++] = {latency_ns, index, type};
    }

    void printReport() const
    {
        if (count == 0) return;

        uint64_t spikes_over_1us = 0;
        uint64_t spikes_over_10us = 0;
        uint64_t spikes_over_100us = 0;

        std::vector<uint64_t> vals; vals.reserve(count);
        uint64_t sum = 0;
        for (size_t i = 0; i < count; ++i)
        {
            uint64_t v = samples[i].latency_ns;
            vals.push_back(v);
            sum += v;
            if (v > 1000) spikes_over_1us++;
            if (v > 10000) spikes_over_10us++;
            if (v > 100000) spikes_over_100us++;
        }

        double mean = static_cast<double>(sum) / count;
        double var = 0.0;
        for (auto v : vals) { double d = static_cast<double>(v) - mean; var += d * d; }
        var /= count;
        double stddev = std::sqrt(var);

        std::sort(vals.begin(), vals.end());
        auto pct = [&](double p)->uint64_t {
            size_t idx = static_cast<size_t>(std::ceil((p / 100.0) * count)) - 1;
            if (idx >= vals.size()) idx = vals.size() - 1;
            return vals[idx];
        };

        uint64_t p50 = pct(50.0);
        uint64_t p90 = pct(90.0);
        uint64_t p99 = pct(99.0);
        uint64_t p999 = pct(99.9);

        std::cout << "\n========== DETAILED LATENCY REPORT ==========" << std::endl;
        std::cout << "Samples: " << count << "\n";
        std::cout << "Mean: " << static_cast<uint64_t>(mean) << " ns  StdDev: " << static_cast<uint64_t>(stddev) << " ns\n";
        std::cout << "P50: " << p50 << " ns  P90: " << p90 << " ns  P99: " << p99 << " ns  P99.9: " << p999 << " ns\n";
        std::cout << "Spikes >1us: " << spikes_over_1us << "  >10us: " << spikes_over_10us << "  >100us: " << spikes_over_100us << "\n";

        // Top spikes
        std::vector<size_t> idxs(count);
        for (size_t i = 0; i < count; ++i) idxs[i] = i;
        std::partial_sort(idxs.begin(), idxs.begin() + std::min<size_t>(10, idxs.size()), idxs.end(),
            [&](size_t a, size_t b) { return samples[a].latency_ns > samples[b].latency_ns; });

        std::cout << "Top spikes (latency ns, sample index, msg type):\n";
        for (size_t i = 0; i < std::min<size_t>(10, idxs.size()); ++i)
        {
            const auto &s = samples[idxs[i]];
            std::cout << "  " << s.latency_ns << " ns, idx=" << s.index << ", type=" << s.type << "\n";
        }

        std::cout << "============================================" << std::endl;
    }
};

static LatencyRecorder g_latency_recorder;

class OrderBook
{
    // --------------------------------------------------------------------
    // 1. PRICE LEVEL VIEW (Aggregated quantity per price)
    // --------------------------------------------------------------------
    // bids[price] = total bid quantity at that price
    // asks[price] = total ask quantity at that price
    std::array<uint32_t, MAX_PRICE + 1> bids{};
    std::array<uint32_t, MAX_PRICE + 1> asks{};

    // Track current best bid and best ask
    uint64_t max_bid_price = 0;
    uint64_t min_ask_price = MAX_PRICE + 1;

    // BITMAP OPTIMIZATION: Eliminate branch misprediction in best bid/ask search
    // Each bit represents a price level that has orders
    // __builtin_ctzll finds the first set bit in O(1) / 1 clock cycle
    static constexpr size_t BITMAP_SIZE = (MAX_PRICE + 64) / 64;
    std::array<uint64_t, BITMAP_SIZE> bid_bitmap{};
    std::array<uint64_t, BITMAP_SIZE> ask_bitmap{};
    
    // Helper: Set bit for price level (mark as occupied)
    inline void set_bid_level(uint64_t price)
    {
        size_t idx = price / 64;
        size_t bit = price % 64;
        bid_bitmap[idx] |= (1ULL << bit);
    }
    
    inline void set_ask_level(uint64_t price)
    {
        size_t idx = price / 64;
        size_t bit = price % 64;
        ask_bitmap[idx] |= (1ULL << bit);
    }
    
    // Helper: Clear bit for price level (mark as empty)
    inline void clear_bid_level(uint64_t price)
    {
        size_t idx = price / 64;
        size_t bit = price % 64;
        bid_bitmap[idx] &= ~(1ULL << bit);
    }
    
    inline void clear_ask_level(uint64_t price)
    {
        size_t idx = price / 64;
        size_t bit = price % 64;
        ask_bitmap[idx] &= ~(1ULL << bit);
    }
    
    // Helper: Find next best ask (lowest price with orders)
    // Uses __builtin_ctzll for O(1) bit scanning
    inline uint64_t find_next_ask(uint64_t start_price)
    {
        for (uint64_t price = start_price; price <= MAX_PRICE; price++)
        {
            size_t idx = price / 64;
            size_t bit = price % 64;
            
            // Mask off bits below current price
            uint64_t masked = ask_bitmap[idx] & (~0ULL << bit);
            if (masked)
            {
                // Found a set bit in this block
                return idx * 64 + __builtin_ctzll(masked);
            }
            
            // Move to next block, check all bits
            for (size_t i = idx + 1; i < BITMAP_SIZE; i++)
            {
                if (ask_bitmap[i])
                {
                    return i * 64 + __builtin_ctzll(ask_bitmap[i]);
                }
            }
            return MAX_PRICE + 1; // No price level found
        }
        return MAX_PRICE + 1;
    }
    
    // Helper: Find next best bid (highest price with orders)
    // Scans downward, uses __builtin_clzll for highest set bit
    inline uint64_t find_next_bid(uint64_t start_price)
    {
        for (int64_t price = start_price; price >= 0; price--)
        {
            size_t idx = price / 64;
            size_t bit = price % 64;
            
            // Mask off bits above current price
            uint64_t masked = bid_bitmap[idx] & ((1ULL << (bit + 1)) - 1);
            if (masked)
            {
                // Found a set bit in this block: locate highest set bit
                return idx * 64 + (63 - __builtin_clzll(masked));
            }
            
            // Move to previous block, check all bits
            for (int64_t i = idx - 1; i >= 0; i--)
            {
                if (bid_bitmap[i])
                {
                    return i * 64 + (63 - __builtin_clzll(bid_bitmap[i]));
                }
            }
            return 0; // No price level found
        }
        return 0;
    }

    // --------------------------------------------------------------------
    // 2. ORDER LOOKUP TABLE (OrderID -> OrderInfo)
    // --------------------------------------------------------------------
    struct OrderInfo
    {
        uint64_t price;
        uint32_t quantity;
        char side; // 'B' = Buy, 'S' = Sell
    };

    // Direct indexing by Order ID (O(1))
    std::array<OrderInfo, 1000001> order_lookup{};

    // Total traded volume accumulator
    uint64_t total_traded_volume;

public:
    OrderBook()
    {
        // All arrays are pre-allocated at compile-time with std::array
        // std::array provides zero-initialization with {} syntax
        total_traded_volume = 0;
    }

    // --------------------------------------------------------------------
    // NEW ORDER ('N')
    // --------------------------------------------------------------------
    void addOrder(uint64_t id, char side, uint64_t price, uint32_t quantity)
    {
        // Store order details for future lookup
        order_lookup[id] = {price, quantity, side};

        // Update aggregated price level
        if (side == 'B')
        {
            bids[price] += quantity;
            set_bid_level(price);  // Mark price level as occupied in bitmap
            max_bid_price = std::max(price, max_bid_price);
        }
        else
        {
            asks[price] += quantity;
            set_ask_level(price);  // Mark price level as occupied in bitmap
            min_ask_price = std::min(price, min_ask_price);
        }
        
        // Publish to ring buffer (fast rdtsc, no CPUID overhead)
        UpdateMessage msg{ id, price, get_timestamp_ns(), quantity, 'N', side };
        updateBuffer.push(msg);
    }

    // --------------------------------------------------------------------
    // CANCEL ORDER ('X')
    // --------------------------------------------------------------------
    void cancelOrder(uint64_t id)
    {
        // Validate ID range
        if (id >= order_lookup.size())
            return;

        OrderInfo &info = order_lookup[id];

        // Ignore if already removed
        if (info.quantity == 0)
            return;

        // Remove quantity from price level
        if (info.side == 'B')
        {
            bids[info.price] -= info.quantity;
            
            // If price level is now empty, clear the bitmap bit (no branch misprediction)
            if (bids[info.price] == 0)
            {
                clear_bid_level(info.price);
                
                // Update best bid only if we cleared the current best price
                if (info.price == max_bid_price)
                {
                    max_bid_price = find_next_bid(max_bid_price - 1);
                }
            }
        }
        else
        {
            asks[info.price] -= info.quantity;
            
            // If price level is now empty, clear the bitmap bit
            if (asks[info.price] == 0)
            {
                clear_ask_level(info.price);
                
                // Update best ask only if we cleared the current best price
                if (info.price == min_ask_price)
                {
                    min_ask_price = find_next_ask(min_ask_price + 1);
                }
            }
        }

        // Publish to ring buffer (fast rdtsc)
        UpdateMessage msg{ id, info.price, get_timestamp_ns(), info.quantity, 'X', info.side };
        updateBuffer.push(msg);
        
        // Mark order as removed
        info.quantity = 0;
    }

    // --------------------------------------------------------------------
    // TRADE ('T')
    // --------------------------------------------------------------------
    // Reduces quantity of both matched orders
    void executeTrade(uint64_t buy_id, uint64_t sell_id, uint32_t qty)
    {
        if (order_lookup[buy_id].quantity > 0 &&
            order_lookup[sell_id].quantity > 0)
        {
            // ----- BUY SIDE -----
            OrderInfo &buy_info = order_lookup[buy_id];
            bids[buy_info.price] -= qty;
            buy_info.quantity -= qty;

            if (bids[buy_info.price] == 0)
            {
                clear_bid_level(buy_info.price);
                if (buy_info.price == max_bid_price)
                {
                    max_bid_price = find_next_bid(max_bid_price - 1);
                }
            }

            // ----- SELL SIDE -----
            OrderInfo &sell_info = order_lookup[sell_id];
            asks[sell_info.price] -= qty;
            sell_info.quantity -= qty;

            if (asks[sell_info.price] == 0)
            {
                clear_ask_level(sell_info.price);
                if (sell_info.price == min_ask_price)
                {
                    min_ask_price = find_next_ask(min_ask_price + 1);
                }
            }

            total_traded_volume += qty;
            
            // Publish to ring buffer (fast rdtsc ~2-4ns)
            UpdateMessage msg{ buy_id, buy_info.price, get_timestamp_ns(), qty, 'T', 'B' };
            updateBuffer.push(msg);
        }
    }
    
    // --------------------------------------------------------------------
    // MODIFY ORDER ('M') - Smart priority preservation
    // --------------------------------------------------------------------
    void modifyOrder(uint64_t id, char new_side, uint64_t new_price, uint32_t new_quantity)
    {
        // Validate ID range
        if (id >= order_lookup.size())
            return;
            
        OrderInfo &info = order_lookup[id];
        
        // Ignore if order doesn't exist
        if (info.quantity == 0)
            return;
        
        uint64_t old_price = info.price;
        uint32_t old_quantity = info.quantity;
        char old_side = info.side;
        
        // CASE 1: Price changed OR Side changed -> Lose priority (Cancel + Add)
        if (new_price != old_price || new_side != old_side)
        {
            cancelOrder(id);
            addOrder(id, new_side, new_price, new_quantity);
            return;
        }
        
        // CASE 2: Quantity increased -> Lose priority (Cancel + Add)
        if (new_quantity > old_quantity)
        {
            cancelOrder(id);
            addOrder(id, new_side, new_price, new_quantity);
            return;
        }
        
        // CASE 3: Quantity decreased -> KEEP PRIORITY (update in place)
        // This is the critical path for maintaining queue position!
        if (new_quantity < old_quantity)
        {
            uint32_t qty_decrease = old_quantity - new_quantity;
            
            // Update price level
            if (info.side == 'B')
            {
                bids[info.price] -= qty_decrease;
                
                // Recompute best bid using bitmap if this level is now empty
                if (bids[info.price] == 0)
                {
                    clear_bid_level(info.price);
                    if (info.price == max_bid_price)
                    {
                        max_bid_price = find_next_bid(max_bid_price - 1);
                    }
                }
            }
            else
            {
                asks[info.price] -= qty_decrease;
                
                // Recompute best ask using bitmap if this level is now empty
                if (asks[info.price] == 0)
                {
                    clear_ask_level(info.price);
                    if (info.price == min_ask_price)
                    {
                        min_ask_price = find_next_ask(min_ask_price + 1);
                    }
                }
            }
            
            // Update order quantity in place - maintains queue position!
            info.quantity = new_quantity;
            
            // Publish to ring buffer (fast rdtsc)
            UpdateMessage msg{ id, new_price, get_timestamp_ns(), new_quantity, 'M', new_side };
            updateBuffer.push(msg);
        }
        
        // CASE 4: Quantity unchanged -> No-op

    }
    
    // pre-fault memory
    void pre_fault_memory()
    {
        // Touch one element per 4KB page to force OS to allocate physical memory
        // This eliminates page faults during hot path execution
        const size_t page_size = 4096;
        
        // Stride = elements per page (no +1! That skips pages)
        for (size_t i = 0; i < order_lookup.size(); i += page_size / sizeof(OrderInfo)) {
            order_lookup[i].quantity = 0; 
        }
        
        for (size_t i = 0; i < bids.size(); i += page_size / sizeof(uint32_t)) {
            bids[i] = 0;
            asks[i] = 0;
        }
        
        for (size_t i = 0; i < bid_bitmap.size(); i += page_size / sizeof(uint64_t)) {
            bid_bitmap[i] = 0;
            ask_bitmap[i] = 0;
        }
    }

    // --------------------------------------------------------------------
    // MARKET ORDER - Crosses the spread and walks the book
    // Bitmap-optimized to eliminate branch misprediction in price level scan
    // --------------------------------------------------------------------
    void executeMarketOrder(char side, uint32_t quantity)
    {
        if (side == 'B') // Market Buy: match against asks
        {
            uint32_t remaining = quantity;
            
            // Walk up the ask side (starting from best ask)
            for (uint64_t price = min_ask_price; price <= MAX_PRICE && remaining > 0; price++)
            {
                if (asks[price] == 0)
                    continue;
                
                uint32_t available = asks[price];
                uint32_t matched = std::min(remaining, available);
                
                // Execute the trade
                asks[price] -= matched;
                remaining -= matched;
                total_traded_volume += matched;
                
                // Publish trade (fast rdtsc ~2-4ns)
                UpdateMessage msg{ 0, price, get_timestamp_ns(), matched, 'T', 'B' };
                updateBuffer.push(msg);
                
                // Update best ask if this level is depleted
                if (asks[price] == 0)
                {
                    clear_ask_level(price);
                    if (price == min_ask_price)
                    {
                        min_ask_price = find_next_ask(min_ask_price + 1);
                    }
                }
            }
            
            if (remaining > 0)
            {
                // Market order couldn't be fully filled - rare but possible
                std::cout << "[WARNING] Market Buy partially filled. Unfilled: " << remaining << "\n";
            }
        }
        else // Market Sell: match against bids
        {
            uint32_t remaining = quantity;
            
            // Walk down the bid side (starting from best bid)
            for (uint64_t price = max_bid_price; price > 0 && remaining > 0; price--)
            {
                if (bids[price] == 0)
                    continue;
                
                uint32_t available = bids[price];
                uint32_t matched = std::min(remaining, available);
                
                // Execute the trade
                bids[price] -= matched;
                remaining -= matched;
                total_traded_volume += matched;
                
                // Publish trade (fast rdtsc ~2-4ns)
                UpdateMessage msg{ 0, price, get_timestamp_ns(), matched, 'T', 'S' };
                updateBuffer.push(msg);
                
                // Update best bid if this level is depleted
                if (bids[price] == 0)
                {
                    clear_bid_level(price);
                    if (price == max_bid_price)
                    {
                        max_bid_price = find_next_bid(max_bid_price - 1);
                    }
                }
            }
            
            if (remaining > 0)
            {
                std::cout << "[WARNING] Market Sell partially filled. Unfilled: " << remaining << "\n";
            }
        }
    }
    
    // Print best bid/ask snapshot
    void printTopOfBook()
    {
        std::cout << "   [BOOK] Bid: "
                  << (max_bid_price > 0 ? bids[max_bid_price] : 0)
                  << " @ " << max_bid_price
                  << " | Ask: "
                  << (min_ask_price <= MAX_PRICE ? asks[min_ask_price] : 0)
                  << " @ " << min_ask_price << "\n";
    }

    uint64_t getTotalTradedVolume() const { return total_traded_volume; }
    size_t getBidsSize() const { return bids.size(); }
    size_t getAsksSize() const { return asks.size(); }
};

OrderBook orderBook;

// --------------------------------------------------------------------
// NETWORK THREAD - Zero-copy packet producer (simulates DMA NIC)
// --------------------------------------------------------------------
// Production HFT: NIC writes packets directly to ring buffer via DMA
// Engine reads pointer, never copies payload → eliminates memcpy overhead
void networkThread(char* file_memory, size_t file_size)
{
    // Pin to Core 0
    pinThreadToCore(0);
    
    std::cout << "[NETWORK] Thread started (zero-copy mode)\n";
    
    size_t offset = 0;
    
    while (offset < file_size && !stopNetworkThread.load(std::memory_order_acquire))
    {
        // Read header to get message length
        StreamHeader *header = reinterpret_cast<StreamHeader *>(file_memory + offset);
        
        // Create zero-copy packet view (no memcpy!)
        PacketView view;
        view.msg_type = *(file_memory + offset + sizeof(StreamHeader)); // Peek at first byte
        view.payload = file_memory + offset + sizeof(StreamHeader);      // Point to data
        
        // Push lightweight view to queue (16 bytes vs 136 bytes)
        while (!inputQueue.push(view))
        {
            // Queue full, busy-wait (no context switch!)
            cpu_pause();
        }
        
        offset += sizeof(StreamHeader) + header->msg_len;
    }
    
    // Signal engine thread that we're done producing
    stopEngine.store(true, std::memory_order_release);
    std::cout << "[NETWORK] Thread stopping. All messages produced\n";
}

// --------------------------------------------------------------------
// PUBLISHER THREAD - Reads from ring buffer and publishes updates
// --------------------------------------------------------------------
void publisherThread()
{
    // Pin to Core 3
    pinThreadToCore(3);
    
    UpdateMessage msg;
    uint64_t published_count = 0;
    
    std::cout << "[PUBLISHER] Thread started\n";
    
    while (!stopPublisher.load(std::memory_order_acquire))
    {
        if (updateBuffer.pop(msg))
        {
            // In a real system, this would publish to:
            // - Market data feed
            // - Client connections
            // - Logging system
            // - Persistence layer
            
            // For now, we just count published messages
            published_count++;
            
            // Optional: Uncomment to see published updates
            // std::cout << "[PUB] Type=" << msg.type 
            //           << " ID=" << msg.order_id 
            //           << " Price=" << msg.price 
            //           << " Qty=" << msg.quantity << "\n";
        }
        else
        {
            // Buffer empty, busy-wait (HFT: never yield the core!)
            cpu_pause();
        }
    }
    
    // Drain remaining messages
    while (updateBuffer.pop(msg))
    {
        published_count++;
    }
    
    std::cout << "[PUBLISHER] Thread stopping. Published " << published_count << " updates\n";
}

int main()
{
    const char *filepath = "market_data.bin";

    // --------------------------------------------------------------------
    // 1. Open file
    // --------------------------------------------------------------------
    int fd = open(filepath, O_RDONLY);
    if (fd == -1)
    {
        perror("open");
        return 1;
    }

    // --------------------------------------------------------------------
    // 2. Get file size (required for mmap)
    // --------------------------------------------------------------------
    struct stat sb;
    if (fstat(fd, &sb) == -1)
    {
        perror("fstat");
        return 1;
    }

    // --------------------------------------------------------------------
    // 3. Memory map file (zero-copy read)
    // --------------------------------------------------------------------
    // PROT_READ  : read-only mapping
    // MAP_PRIVATE: copy-on-write (no disk modification)
    char *file_memory =
        static_cast<char *>(mmap(nullptr, sb.st_size,
                                 PROT_READ, MAP_PRIVATE, fd, 0));

    if (file_memory == MAP_FAILED)
    {
        perror("mmap");
        return 1;
    }

    // --------------------------------------------------------------------
    // 4. Start threads and warmup
    // --------------------------------------------------------------------
    
    // Calibrate CPU frequency for accurate cycle-to-nanosecond conversion
    std::cout << "Calibrating CPU frequency...\n";
    g_cycles_per_ns = calibrate_cpu_frequency();
    std::cout << "CPU Frequency: " << std::fixed << std::setprecision(2) 
              << (g_cycles_per_ns * 1000.0) << " MHz\n\n";
    
    // Start publisher thread
    std::thread publisher(publisherThread);
    
    // Start network thread (producer)
    std::thread network(networkThread, file_memory, sb.st_size);
    
    // Pin engine thread to Core 2 (isolated core for best performance)
    pinThreadToCore(2);
    
    std::cout << "\n========== SYSTEM CONFIGURATION ==========\n";
    std::cout << "Engine Thread:    Core 2 " << (isCoreIsolated(2) ? "(ISOLATED ✓)" : "(NOT ISOLATED ✗)") << "\n";
    std::cout << "Network Thread:   Core 0\n";
    std::cout << "Publisher Thread: Core 3\n";
    std::cout << "Timing Method:    RDTSC (hardware TSC)\n";
    
    if (!isCoreIsolated(2))
    {
        std::cout << "\n⚠️  WARNING: Core 2 is NOT isolated!\n";
        std::cout << "    Expected jitter: 400+ μs\n";
        std::cout << "    To isolate: Add to GRUB config:\n";
        std::cout << "    isolcpus=2 nohz_full=2 rcu_nocbs=2\n";
    }
    
    std::cout << "==========================================\n\n";
    
    updateBuffer.pre_fault_memory();
    inputQueue.pre_fault_memory();
    orderBook.pre_fault_memory();

    // Warmup phase to prime CPU caches and branch predictors
    std::cout << "Warming up caches...\n";
    for (int i = 0; i < 10000; i++)
    {
        orderBook.addOrder(i, 'B', i % 1000, 10);
        orderBook.cancelOrder(i);
    }
    std::cout << "Warmup complete. Starting benchmark.\n";

    // --------------------------------------------------------------------
    // 5. Engine loop - Process messages from input queue
    // --------------------------------------------------------------------
    uint32_t messageCount = 0;
    LatencyHistogram histogram(10); // 10ns buckets
    
    // TIMING: Intel's recommended RDTSC sandwich pattern
    // Measurement boundaries:
    //   - Start: CPUID + RDTSC (fully serializing, ~20-30ns)
    //   - End: RDTSCP + LFENCE (ensures completion, ~8-10ns)
    // Hot path timestamps:
    //   - get_timestamp_ns() uses fast rdtsc (no CPUID, ~2-4ns)
    //   - CPUID only at measurement boundaries, NOT in order book methods
    
    uint64_t totalStart = rdtsc_start();

    // Process until network thread signals completion and queue is drained
    // ZERO-COPY: Cast directly from pointer, eliminate memcpy overhead
    while (true)
    {
        PacketView view;
        
        // Try to pop from input queue (lock-free)
        if (!inputQueue.pop(view))
        {
            if (stopEngine.load(std::memory_order_acquire))
            {
                // Double-check to ensure nothing slipped in before the flag was raised
                if (!inputQueue.pop(view)) 
                {
                    break; 
                }
            }
            else 
            {
                cpu_pause();
                continue;
            }
        }
        
        uint64_t msgStart = rdtsc_start();

        // Cast directly from wire memory (zero copies!)
        // In production: payload points to DMA buffer written by NIC hardware
        switch (view.msg_type)
        {
        case 'T':
        {
            const TradeMessage *t = reinterpret_cast<const TradeMessage *>(view.payload);
            orderBook.executeTrade(
                t->buy_order_id,
                t->sell_order_id,
                t->quantity);
            break;
        }
        case 'N':
        {
            const OrderMessage *o = reinterpret_cast<const OrderMessage *>(view.payload);
            orderBook.addOrder(
                o->order_id,
                o->side,
                o->price,
                o->quantity);
            break;
        }
        case 'X':
        {
            const OrderMessage *o = reinterpret_cast<const OrderMessage *>(view.payload);
            orderBook.cancelOrder(o->order_id);
            break;
        }
        case 'M': // Modify with smart priority preservation
        {
            const OrderMessage *o = reinterpret_cast<const OrderMessage *>(view.payload);
            orderBook.modifyOrder(
                o->order_id,
                o->side,
                o->price,
                o->quantity);
            break;
        }
        case 'K': // Market Order (new message type)
        {
            const OrderMessage *o = reinterpret_cast<const OrderMessage *>(view.payload);
            orderBook.executeMarketOrder(o->side, o->quantity);
            break;
        }
        }

        uint64_t msgEnd = rdtsc_end();

        uint64_t latency_cycles = msgEnd - msgStart;
        uint64_t latency_ns = cycles_to_ns(latency_cycles);
        histogram.addSample(latency_ns);
        // Record detailed sample for post-run analysis (store msg type and index)
        g_latency_recorder.addSample(latency_ns, view.msg_type, messageCount);

        messageCount++;
    }

    uint64_t totalEnd = rdtsc_end();
    uint64_t totalDuration_us = cycles_to_ns(totalEnd - totalStart) / 1000;

    // --------------------------------------------------------------------
    // Performance Statistics
    // --------------------------------------------------------------------
    std::cout << "\n================ PERFORMANCE SUMMARY ================\n";
    std::cout << "Processed messages: " << messageCount << "\n";
    std::cout << "Total time: " << totalDuration_us << " us\n";
    std::cout << "Throughput: " << (messageCount * 1000000.0 / totalDuration_us) << " msgs/sec\n";
    std::cout << "Timing Method: CPUID+RDTSC / RDTSCP+LFENCE (Intel sandwich)\n";
    std::cout << "CPU Frequency: " << std::fixed << std::setprecision(2) 
              << (g_cycles_per_ns * 1000.0) << " MHz\n";
    std::cout << "====================================================\n";
    
    // Print histogram and jitter analysis
    histogram.printHistogram();
    histogram.printJitterAnalysis();
    // Print detailed latency recorder report
    g_latency_recorder.printReport();

    std::cout << "Final Book Size -> Bids: "
              << orderBook.getBidsSize()
              << " | Asks: "
              << orderBook.getAsksSize() << "\n";

    std::cout << "\nTotal Traded Volume = "
              << orderBook.getTotalTradedVolume() << "\n";

    // Stop threads
    std::cout << "\nStopping threads...\n";
    stopNetworkThread.store(true, std::memory_order_release);
    network.join();
    std::cout << "Network thread stopped.\n";
    
    stopPublisher.store(true, std::memory_order_release);
    publisher.join();
    std::cout << "Publisher thread stopped.\n";

    // Cleanup memory mapping
    munmap(file_memory, sb.st_size);
    close(fd);

    return 0;
}
