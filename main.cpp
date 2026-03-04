#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "common.h"
#include "timing.h"
#include "cpu_utils.h"
#include "messages.h"
#include "order_book.h"
#include "latency.h"
#include "threads.h"
#include "journal.h"

// Global order book (large static allocation -- no heap pressure at runtime)
static OrderBook orderBook;

// Initialize the persistent journal (Sequential Message Log)
// 10M entries = ~500MB on disk for 48-byte UpdateMessages
MappedJournal<UpdateMessage, 10000000> persistentJournal("event_journal.bin");

int main()
{
    const char* filepath = "market_data.bin";

    // ----------------------------------------------------------------
    // 1. Open and memory-map the market data file
    // ----------------------------------------------------------------
    int fd = open(filepath, O_RDONLY);
    if (fd == -1) { perror("open"); return 1; }

    struct stat sb;
    if (fstat(fd, &sb) == -1) { perror("fstat"); return 1; }

    // PROT_READ + MAP_PRIVATE: read-only, copy-on-write (no disk writes)
    char* file_memory = static_cast<char*>(
        mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (file_memory == MAP_FAILED) { perror("mmap"); return 1; }

    // ----------------------------------------------------------------
    // 2. Calibrate TSC -> nanosecond conversion
    // ----------------------------------------------------------------
    std::cout << "Calibrating CPU frequency...\n";
    g_cycles_per_ns = calibrate_cpu_frequency();
    std::cout << "CPU Frequency: " << std::fixed << std::setprecision(2)
              << (g_cycles_per_ns * 1000.0) << " MHz\n\n";

    // ----------------------------------------------------------------
    // 3. Launch background threads
    // ----------------------------------------------------------------
    std::thread publisher(publisherThread);
    std::thread network(networkThread, file_memory, sb.st_size);

    // Pin the engine (this thread) to Core 2 -- the isolated hot-path core
    pinThreadToCore(2);

    std::cout << "\n========== SYSTEM CONFIGURATION ==========\n";
    std::cout << "Engine Thread:    Core 2 "
              << (isCoreIsolated(2) ? "(ISOLATED [OK])" : "(NOT ISOLATED [!!])") << "\n";
    std::cout << "Network Thread:   Core 0\n";
    std::cout << "Publisher Thread: Core 3\n";
    std::cout << "Timing Method:    RDTSC (hardware TSC)\n";

    if (!isCoreIsolated(2))
    {
        std::cout << "\n[!!]  WARNING: Core 2 is NOT isolated!\n";
        std::cout << "    Expected jitter: 400+ us\n";
        std::cout << "    To isolate, add to GRUB config:\n";
        std::cout << "    isolcpus=2 nohz_full=2 rcu_nocbs=2\n";
    }
    std::cout << "==========================================\n\n";

    // ----------------------------------------------------------------
    // 4. Pre-fault all hot-path memory to eliminate runtime page faults
    // ----------------------------------------------------------------
    updateBuffer.pre_fault_memory();
    inputQueue.pre_fault_memory();
    orderBook.pre_fault_memory();

    // ----------------------------------------------------------------
    // 5. Warmup -- prime CPU caches and branch predictors
    // ----------------------------------------------------------------
    std::cout << "Warming up caches...\n";
    for (int i = 0; i < 10000; i++)
    {
        // participant_id cycles across five owners to warm STP branch predictors.
        orderBook.addOrder(i, 'B', i % 1000, 10, static_cast<uint32_t>(i % 5));
        orderBook.cancelOrder(i);
    }
    std::cout << "Warmup complete. Starting benchmark.\n\n";

    // ----------------------------------------------------------------
    // 6. Engine loop
    // ----------------------------------------------------------------
    // Timing: Intel-recommended RDTSC sandwich
    //   Measurement boundary start : CPUID + RDTSC  (~20-30 ns overhead)
    //   Measurement boundary end   : RDTSCP + LFENCE (~8-10 ns overhead)
    //   Hot-path timestamps        : rdtsc_fast (no serialisation, ~2-4 ns)

    uint32_t messageCount = 0;

    // ----------------------------------------------------------------
    // Pre-scan: count total messages so we can size the TSC buffer exactly.
    // This eliminates the conditional branch inside the hot loop that was
    // causing two problems:
    //   1. First N messages paid branch + store cost; rest paid almost nothing
    //   2. Latency samples only covered the first N messages, not all of them
    // One extra pass over the (already mmap'd) file is cheap vs the distortion.
    // ----------------------------------------------------------------
    uint32_t totalMessages = 0;
    {
        size_t offset = 0;
        while (offset < static_cast<size_t>(sb.st_size))
        {
            const StreamHeader* h = reinterpret_cast<const StreamHeader*>(file_memory + offset);
            totalMessages++;
            offset += sizeof(StreamHeader) + h->msg_len;
        }
    }
    std::cout << "Total messages in file: " << totalMessages << "\n";

    // Allocate exact-fit buffers on the heap (may be millions of entries)
    std::vector<uint64_t> tsc_start_buf(totalMessages);
    std::vector<uint64_t> tsc_end_buf(totalMessages);
    std::vector<char>     msg_type_buf(totalMessages);
    
    LatencyHistogram histogram(10);  // Initialize before hot loop

    uint64_t totalStart = rdtsc_start();

    // Fire the starting pistol: network thread was blocked waiting for this.
    // Data injection now begins AFTER the benchmark timer has started, so
    // every message experiences real inter-core queue contention.
    startNetworkTraffic.store(true, std::memory_order_release);

    while (true)
    {
        PacketView view;

        if (!inputQueue.pop(view))
        {
            if (stopEngine.load(std::memory_order_acquire))
            {
                if (!inputQueue.pop(view))
                    break;
            }
            else
            {
                cpu_pause();
                continue;
            }
        }

        uint64_t msgStart = rdtsc_start(); // LFENCE + RDTSC (~20 cycles)

        switch (view.msg_type)
        {
        case 'N':
        {
            const auto* o = reinterpret_cast<const OrderMessage*>(view.payload);
            orderBook.addOrder(o->order_id, o->side, o->price, o->quantity);
            break;
        }
        case 'X':
        {
            const auto* o = reinterpret_cast<const OrderMessage*>(view.payload);
            orderBook.cancelOrder(o->order_id);
            break;
        }
        case 'T':
        {
            const auto* t = reinterpret_cast<const TradeMessage*>(view.payload);
            orderBook.executeTrade(t->buy_order_id, t->sell_order_id, t->quantity);
            break;
        }
        case 'M':
        {
            const auto* o = reinterpret_cast<const OrderMessage*>(view.payload);
            orderBook.modifyOrder(o->order_id, o->side, o->price, o->quantity);
            break;
        }
        case 'K':
        {
            const auto* o = reinterpret_cast<const OrderMessage*>(view.payload);
            orderBook.executeMarketOrder(o->side, o->quantity);
            break;
        }
        }

        // Unconditional stores -- no branch, every message recorded
        tsc_start_buf[messageCount] = msgStart;
        tsc_end_buf[messageCount]   = rdtsc_end(); // RDTSCP + LFENCE
        msg_type_buf[messageCount]  = view.msg_type;
        
        // Calculate latency and record it (matches original engine.cpp hot loop)
        uint64_t latency_ns = cycles_to_ns(tsc_end_buf[messageCount] - tsc_start_buf[messageCount]);
        histogram.addSample(latency_ns);
        g_latency_recorder.addSample(latency_ns, view.msg_type, messageCount);

        messageCount++;
    }

    uint64_t totalEnd         = rdtsc_end();
    uint64_t totalDuration_us = cycles_to_ns(totalEnd - totalStart) / 1000;

    // ----------------------------------------------------------------
    // 7. Performance report
    // ----------------------------------------------------------------
    std::cout << "\n================ PERFORMANCE SUMMARY ================\n";
    std::cout << "Processed messages: " << messageCount << "\n";
    std::cout << "Total time:         " << totalDuration_us << " us\n";
    std::cout << "Throughput:         "
              << (messageCount * 1000000.0 / totalDuration_us) << " msgs/sec\n";
    std::cout << "Timing method:      CPUID+RDTSC / RDTSCP+LFENCE\n";
    std::cout << "CPU Frequency:      " << std::fixed << std::setprecision(2)
              << (g_cycles_per_ns * 1000.0) << " MHz\n";
    std::cout << "====================================================\n";

    histogram.printHistogram();
    histogram.printJitterAnalysis();
    g_latency_recorder.printReport();

    std::cout << "Final Book Size -> Bids: " << orderBook.getBidsSize()
              << " | Asks: "                 << orderBook.getAsksSize() << "\n";
    std::cout << "Total Traded Volume = " << orderBook.getTotalTradedVolume() << "\n";
    std::cout << "VWAP = " << std::fixed << std::setprecision(2) << orderBook.getVWAP() << "\n";

    // Save final state snapshot
    if (orderBook.saveSnapshot("orderbook_snapshot.bin")) {
        std::cout << "Snapshot saved to orderbook_snapshot.bin\n";
    }

    // ----------------------------------------------------------------
    // 8. Graceful shutdown
    // ----------------------------------------------------------------
    std::cout << "\nStopping threads...\n";

    stopNetworkThread.store(true, std::memory_order_release);
    network.join();
    std::cout << "Network thread stopped.\n";

    stopPublisher.store(true, std::memory_order_release);
    publisher.join();
    std::cout << "Publisher thread stopped.\n";

    munmap(file_memory, sb.st_size);
    close(fd);

    return 0;
}