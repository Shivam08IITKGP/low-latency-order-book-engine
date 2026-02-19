#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <map> // Previously considered for price levels (RB-tree, O(logN))
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <cstring>
#include <sched.h> // For CPU pinning
#include <pthread.h>
#include <iomanip>
#include <xmmintrin.h> // For _mm_lfence() memory fence intrinsic
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
    char type; // 'N'=New, 'X'=Cancel, 'T'=Trade, 'M'=Modify
    uint64_t order_id;
    uint64_t price;
    uint32_t quantity;
    char side;
    uint64_t timestamp_ns;
};

template<typename T, size_t Size>
class RingBuffer
{
private:
    T buffer[Size];
    std::atomic<size_t> write_idx{0};
    std::atomic<size_t> read_idx{0};
    
public:
    // Try to push a message (returns false if buffer is full)
    bool push(const T& msg)
    {
        size_t current_write = write_idx.load(std::memory_order_relaxed);
        size_t next_write = (current_write + 1) % Size;
        
        // Check if buffer is full
        if (next_write == read_idx.load(std::memory_order_acquire))
            return false;
        
        buffer[current_write] = msg;
        write_idx.store(next_write, std::memory_order_release);
        return true;
    }
    
    // Try to pop a message (returns false if buffer is empty)
    bool pop(T& msg)
    {
        size_t current_read = read_idx.load(std::memory_order_relaxed);
        
        // Check if buffer is empty
        if (current_read == write_idx.load(std::memory_order_acquire))
            return false;
        
        msg = buffer[current_read];
        read_idx.store((current_read + 1) % Size, std::memory_order_release);
        return true;
    }
    
    bool isEmpty() const
    {
        return read_idx.load(std::memory_order_acquire) == 
               write_idx.load(std::memory_order_acquire);
    }
};

// Global ring buffer - 1M slots should handle high throughput
RingBuffer<UpdateMessage, 1048576> updateBuffer;
std::atomic<bool> stopPublisher{false};

// --------------------------------------------------------------------
// SPSC INPUT QUEUE for lock-free communication between threads
// --------------------------------------------------------------------
struct InputMessage
{
    StreamHeader header;
    char payload[128]; // Enough for any message type
};

// Input queue: Network thread produces, Engine thread consumes
RingBuffer<InputMessage, 524288> inputQueue;
std::atomic<bool> stopNetworkThread{false};
std::atomic<bool> stopEngine{false};  // Signal engine to stop after network is done

// --------------------------------------------------------------------
// CPU PINNING UTILITIES
// --------------------------------------------------------------------
bool pinThreadToCore(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    
    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (result != 0)
    {
        std::cerr << "Failed to pin thread to core " << core_id << "\n";
        return false;
    }
    
    std::cout << "[CPU] Thread pinned to core " << core_id << "\n";
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
// RDTSC - Read CPU Time Stamp Counter with Serialization
// --------------------------------------------------------------------
// Prevents out-of-order execution from reordering measurements
// Start: lfence + rdtsc + lfence (~6ns)
// End: rdtscp (~3ns, built-in serialization)
inline uint64_t rdtsc_start()
{
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    _mm_lfence();
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    _mm_lfence();
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
    // RDTSCP: Serializing read - waits for all prior instructions to complete
    // No lfence needed - instruction is inherently serializing
    __asm__ __volatile__ ("rdtscp" : "=a" (lo), "=d" (hi), "=c" (aux));
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

// Get timestamp in nanoseconds (for backward compatibility)
inline uint64_t get_timestamp_ns()
{
    return cycles_to_ns(rdtsc_start());
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

class OrderBook
{
    // --------------------------------------------------------------------
    // 1. PRICE LEVEL VIEW (Aggregated quantity per price)
    // --------------------------------------------------------------------
    // bids[price] = total bid quantity at that price
    // asks[price] = total ask quantity at that price
    std::vector<uint32_t> bids;
    std::vector<uint32_t> asks;

    // Track current best bid and best ask
    uint64_t max_bid_price = 0;
    uint64_t min_ask_price = MAX_PRICE + 1;

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
    std::vector<OrderInfo> order_lookup;

    // Total traded volume accumulator
    uint64_t total_traded_volume;

public:
    OrderBook()
    {
        // Pre-allocate space for up to 1M orders
        order_lookup.resize(1000001);

        // Allocate full price range
        bids.resize(MAX_PRICE + 1);
        asks.resize(MAX_PRICE + 1);

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
            max_bid_price = std::max(price, max_bid_price);
        }
        else
        {
            asks[price] += quantity;
            min_ask_price = std::min(price, min_ask_price);
        }
        
        // Publish to ring buffer (memory-fenced rdtsc)
        UpdateMessage msg{'N', id, price, quantity, side, get_timestamp_ns()};
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

            // Recompute best bid if needed
            if (info.price == max_bid_price && bids[info.price] == 0)
                while (max_bid_price > 0 && bids[max_bid_price] == 0)
                    max_bid_price--;
        }
        else
        {
            asks[info.price] -= info.quantity;

            // Recompute best ask if needed
            if (info.price == min_ask_price && asks[info.price] == 0)
                while (min_ask_price <= MAX_PRICE && asks[min_ask_price] == 0)
                    min_ask_price++;
        }

        // Publish to ring buffer before marking as removed
        UpdateMessage msg{'X', id, info.price, info.quantity, info.side, get_timestamp_ns()};
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

            if (buy_info.price == max_bid_price &&
                bids[buy_info.price] == 0)
                while (max_bid_price > 0 && bids[max_bid_price] == 0)
                    max_bid_price--;

            // ----- SELL SIDE -----
            OrderInfo &sell_info = order_lookup[sell_id];
            asks[sell_info.price] -= qty;
            sell_info.quantity -= qty;

            if (sell_info.price == min_ask_price &&
                asks[sell_info.price] == 0)
                while (min_ask_price <= MAX_PRICE &&
                       asks[min_ask_price] == 0)
                    min_ask_price++;

            total_traded_volume += qty;
            
            // Publish to ring buffer (NOTE: rdtsc overhead ~2-4ns)
            UpdateMessage msg{'T', buy_id, buy_info.price, qty, 'B', get_timestamp_ns()};
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
                
                // Recompute best bid if level is now empty
                if (info.price == max_bid_price && bids[info.price] == 0)
                    while (max_bid_price > 0 && bids[max_bid_price] == 0)
                        max_bid_price--;
            }
            else
            {
                asks[info.price] -= qty_decrease;
                
                // Recompute best ask if level is now empty
                if (info.price == min_ask_price && asks[info.price] == 0)
                    while (min_ask_price <= MAX_PRICE && asks[min_ask_price] == 0)
                        min_ask_price++;
            }
            
            // Update order quantity in place - maintains queue position!
            info.quantity = new_quantity;
            
            // Publish to ring buffer (NOTE: rdtsc overhead ~2-4ns)
            UpdateMessage msg{'M', id, new_price, new_quantity, new_side, get_timestamp_ns()};
            updateBuffer.push(msg);
        }
        
        // CASE 4: Quantity unchanged -> No-op
    }

    // --------------------------------------------------------------------
    // MARKET ORDER - Crosses the spread and walks the book
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
                
                // Publish trade (NOTE: rdtsc overhead ~2-4ns)
                UpdateMessage msg{'T', 0, price, matched, 'B', get_timestamp_ns()};
                updateBuffer.push(msg);
                
                // Update best ask if this level is depleted
                if (asks[price] == 0 && price == min_ask_price)
                {
                    while (min_ask_price <= MAX_PRICE && asks[min_ask_price] == 0)
                        min_ask_price++;
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
                
                // Publish trade (NOTE: rdtsc overhead ~2-4ns)
                UpdateMessage msg{'T', 0, price, matched, 'S', get_timestamp_ns()};
                updateBuffer.push(msg);
                
                // Update best bid if this level is depleted
                if (bids[price] == 0 && price == max_bid_price)
                {
                    while (max_bid_price > 0 && bids[max_bid_price] == 0)
                        max_bid_price--;
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
// NETWORK THREAD - Produces messages (simulates receiving from network)
// --------------------------------------------------------------------
void networkThread(char* file_memory, size_t file_size)
{
    // Pin to Core 0
    pinThreadToCore(0);
    
    std::cout << "[NETWORK] Thread started\n";
    
    size_t offset = 0;
    
    while (offset < file_size && !stopNetworkThread.load(std::memory_order_acquire))
    {
        // Parse message from file
        StreamHeader *header = reinterpret_cast<StreamHeader *>(file_memory + offset);
        size_t msg_total_size = sizeof(StreamHeader) + header->msg_len;
        
        // Create input message
        InputMessage input_msg;
        input_msg.header = *header;
        memcpy(input_msg.payload, file_memory + offset + sizeof(StreamHeader), header->msg_len);
        
        // Push to input queue (lock-free)
        while (!inputQueue.push(input_msg))
        {
            // Queue full, busy-wait (no context switch!)
            cpu_pause();
        }
        
        offset += msg_total_size;
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
    std::cout << "Engine Thread:    Core 2 (isolated)\n";
    std::cout << "Network Thread:   Core 0\n";
    std::cout << "Publisher Thread: Core 3\n";
    std::cout << "Timing Method:    RDTSC (hardware TSC)\n";
    std::cout << "==========================================\n\n";
    
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
    
    // TIMING: RDTSC + RDTSCP for minimal overhead
    // Start: lfence + rdtsc + lfence (~6ns)
    // End: rdtscp (serializing, ~3ns)
    // RDTSCP waits for all instructions to complete before reading TSC
    // Total overhead: ~9ns per message (vs ~20ns for chrono)
    
    uint64_t totalStart = rdtsc_start();

    // Process until network thread signals completion and queue is drained
    while (true)
    {
        InputMessage input_msg;
        
        // Try to pop from input queue (lock-free)
        if (!inputQueue.pop(input_msg))
        {
            if (stopEngine.load(std::memory_order_acquire))
            {
                // Double-check to ensure nothing slipped in before the flag was raised
                if (!inputQueue.pop(input_msg)) 
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

        // First byte of payload is message type
        char msgType = input_msg.payload[0];
        char *payload = input_msg.payload;

        switch (msgType)
        {
        case 'T':
        {
            TradeMessage *t =
                reinterpret_cast<TradeMessage *>(payload);
            orderBook.executeTrade(
                t->buy_order_id,
                t->sell_order_id,
                t->quantity);
            break;
        }
        case 'N':
        {
            OrderMessage *o =
                reinterpret_cast<OrderMessage *>(payload);
            orderBook.addOrder(
                o->order_id,
                o->side,
                o->price,
                o->quantity);
            break;
        }
        case 'X':
        {
            OrderMessage *o =
                reinterpret_cast<OrderMessage *>(payload);
            orderBook.cancelOrder(o->order_id);
            break;
        }
        case 'M': // Modify with smart priority preservation
        {
            OrderMessage *o =
                reinterpret_cast<OrderMessage *>(payload);
            orderBook.modifyOrder(
                o->order_id,
                o->side,
                o->price,
                o->quantity);
            break;
        }
        case 'K': // Market Order (new message type)
        {
            OrderMessage *o =
                reinterpret_cast<OrderMessage *>(payload);
            orderBook.executeMarketOrder(o->side, o->quantity);
            break;
        }
        }

        uint64_t msgEnd = rdtsc_end();

        uint64_t latency_cycles = msgEnd - msgStart;
        uint64_t latency_ns = cycles_to_ns(latency_cycles);
        histogram.addSample(latency_ns);

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
    std::cout << "Timing Method: RDTSC (start) + RDTSCP (end)\n";
    std::cout << "CPU Frequency: " << std::fixed << std::setprecision(2) 
              << (g_cycles_per_ns * 1000.0) << " MHz\n";
    std::cout << "====================================================\n";
    
    // Print histogram and jitter analysis
    histogram.printHistogram();
    histogram.printJitterAnalysis();

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
