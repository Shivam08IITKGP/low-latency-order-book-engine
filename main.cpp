#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <atomic>
#include <thread>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "common.h"
#include "timing.h"
#include "cpu_utils.h"
#include "order_book.h"
#include "messages.h"
#include "threads.h"
#include "latency.h"
#include "journal.h"

// Global instances for inter-thread communication
OrderBook orderBook;
MappedJournal<UpdateMessage, 10000000> persistentJournal("event_journal.bin");

int main(int argc, char* argv[])
{
    // Frequency calibration (TSC based)
    g_cycles_per_ns = calibrate_cpu_frequency();
    std::cout << "CPU Frequency: " << std::fixed << std::setprecision(2) 
              << (g_cycles_per_ns * 1000.0) << " MHz\n\n";

    // Engine thread affinity
    pinThreadToCore(2);

    std::cout << "========== SYSTEM CONFIGURATION ==========\n"
              << "Engine Thread:    Core 2 " << (isCoreIsolated(2) ? "(ISOLATED [OK])" : "[WARNING: NOT ISOLATED]") << "\n"
              << "Network Thread:   Core 0 " << (isCoreIsolated(0) ? "(ISOLATED [OK])" : "[WARNING: NOT ISOLATED]") << "\n"
              << "Publisher Thread: Core 3 " << (isCoreIsolated(3) ? "(ISOLATED [OK])" : "[WARNING: NOT ISOLATED]") << "\n"
              << "==========================================\n\n";

    // Memory mapping of market data
    int fd = open("market_data.bin", O_RDONLY);
    if (fd < 0) { perror("open market_data.bin"); return 1; }

    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;

    // market_data.bin is memory mapped to file memory
    // the entire data stream into the process's virtual address space
    // mmap returns data in void* pointer, so we static_cast it to char* pointer
    // PROT_READ -> Read only
    // MAP_PRIVATE -> Private copy of data
    // fd -> File descriptor
    // 0 -> Offset
    char* file_memory = static_cast<char*>(mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (file_memory == MAP_FAILED) { perror("mmap"); return 1; }

    // Memory pre-faulting to eliminate demand-paging jitter
    std::cout << "Pre-faulting memory...\n";
    orderBook.pre_fault_memory();
    updateBuffer.pre_fault_memory();
    inputQueue.pre_fault_memory();

    // Start background threads
    std::thread pubThread(publisherThread);
    std::thread netThread(networkThread, file_memory, file_size);

    std::cout << "Warming up caches...\n";
    for (int i = 0; i < 10000; i++) cpu_pause();
    
    // Benchmark Execution
    size_t totalMessages = 0;
    size_t offset = 0;
    while (offset < file_size) {
        auto* header = reinterpret_cast<StreamHeader*>(file_memory + offset);
        totalMessages++;
        offset += sizeof(StreamHeader) + header->msg_len;
    }

    std::vector<uint64_t> tsc_start_buf(totalMessages);
    std::vector<uint64_t> tsc_end_buf(totalMessages);
    std::vector<char> msg_type_buf(totalMessages);

    std::cout << "Starting benchmark (" << totalMessages << " messages)...\n";
    uint64_t totalStart = rdtsc_start_full();
    // This contains CPUID wall i.e. CPU Identification
    // strictly serializes the instructions
    // Slow, as it fetches info about CPU

    startNetworkTraffic.store(true, std::memory_order_release);

    size_t processed = 0;
    while (processed < totalMessages) {
        PacketView view;
        if (inputQueue.pop(view)) {
            uint64_t ts_start = rdtsc_start();
            // This rdtsc, contains load fence (LDFENCE)
            // 
            __builtin_prefetch(view.payload, 0 /* read-only */, 3 /* highest urgency */);
            
            const auto* msg = reinterpret_cast<const OrderMessage*>(view.payload);
            if (view.msg_type == 'N') {
                orderBook.addOrder(msg->order_id, msg->side, msg->price, msg->quantity);
            } else if (view.msg_type == 'X') {
                orderBook.handleCancel(msg->order_id);
            }

            uint64_t ts_end = rdtsc_end();
            
            tsc_start_buf[processed] = ts_start;
            tsc_end_buf[processed] = ts_end;
            msg_type_buf[processed] = view.msg_type;
            processed++;
        }
    }
    uint64_t totalEnd = rdtsc_end();

    // Post-processing and Analysis
    std::cout << "Benchmark complete. Analyzing results...\n";
    for (size_t i = 0; i < processed; i++) {
        uint64_t diff = tsc_end_buf[i] - tsc_start_buf[i];
        g_latency_recorder.addSample(cycles_to_ns(diff), msg_type_buf[i], i);
    }

    double totalTimeSec = cycles_to_ns(totalEnd - totalStart) / 1e9;
    std::cout << "\n================ PERFORMANCE SUMMARY ================\n"
              << "Processed messages: " << processed << "\n"
              << "Total time:         " << (totalTimeSec * 1000.0) << " ms\n"
              << "Throughput:         " << (processed / totalTimeSec) << " msgs/sec\n"
              << "====================================================\n";

    g_latency_recorder.printReport();
    // orderBook.printTopOfBook();

    // Cleanup
    stopPublisher.store(true, std::memory_order_release);
    netThread.join();
    pubThread.join();
    munmap(file_memory, file_size);
    close(fd);
    // orderBook.saveSnapshot("orderbook_snapshot.bin");

    return 0;
}