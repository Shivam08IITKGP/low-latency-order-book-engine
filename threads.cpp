#include "threads.h"
#include "messages.h"
#include "cpu_utils.h"
#include "common.h"
#include "journal.h"
#include <iostream>

// Defined in main.cpp alongside the global OrderBook and MappedJournal instances.
extern MappedJournal<UpdateMessage, 10000000> persistentJournal;

// --------------------------------------------------------------------
// NETWORK THREAD -- zero-copy producer (Core 0)
// --------------------------------------------------------------------

void networkThread(char* file_memory, size_t file_size)
{
    pinThreadToCore(0);

    // Wait for main thread to finish warmup before injecting benchmark data.
    // Without this, the network thread pre-fills the entire inputQueue while
    // main is still doing pre_fault_memory() + warmup, so the engine loop
    // drains a hot, fully-populated queue instead of experiencing real
    // inter-core latency.
    while (!startNetworkTraffic.load(std::memory_order_acquire))
        cpu_pause();

    std::cout << "[NETWORK] Thread started (zero-copy mode)\n";

    size_t offset = 0;

    while (offset < file_size && !stopNetworkThread.load(std::memory_order_acquire))
    {
        auto* header = reinterpret_cast<StreamHeader*>(file_memory + offset);

        // Build a zero-copy view: just a pointer + cached type byte
        PacketView view;
        view.msg_type = *(file_memory + offset + sizeof(StreamHeader));
        view.payload  =   file_memory + offset + sizeof(StreamHeader);

        // Busy-wait if the engine hasn't drained the queue yet
        while (!inputQueue.push(view))
            cpu_pause();

        offset += sizeof(StreamHeader) + header->msg_len;
    }

    // Signal the engine that no more packets are coming
    stopEngine.store(true, std::memory_order_release);
    std::cout << "[NETWORK] Thread stopping. All messages produced.\n";
}

// --------------------------------------------------------------------
// PUBLISHER THREAD -- update consumer (Core 3)
// --------------------------------------------------------------------

void publisherThread()
{
    pinThreadToCore(3);
    std::cout << "[PUBLISHER] Thread started\n";

    UpdateMessage msg;
    uint64_t published_count = 0;

    // volatile accumulator: forces the compiler to actually read msg fields
    // from the ring buffer. Without this, LTO can prove the data written by
    // the engine thread is never observed and strip the writes entirely,
    // making addOrder() appear much faster than it really is.
    volatile uint64_t dummy_accumulator = 0;

    while (!stopPublisher.load(std::memory_order_acquire))
    {
        if (updateBuffer.pop(msg))
        {
            // PERSISTENCE: Write every outgoing event directly to the mapped journal.
            // This happens on the Publisher thread (Core 3), offloading the
            // I/O overhead from the Engine thread (Core 2).
            persistentJournal.append(msg);

            dummy_accumulator += msg.price; // must read the payload -- defeats dead-store elimination
            published_count++;
        }
        else
        {
            // Empty -- busy-wait; never yield in HFT (context switch = 1-3 us)
            cpu_pause();
        }
    }

    // Drain any messages that arrived before the stop flag was seen
    while (updateBuffer.pop(msg))
    {
        dummy_accumulator += msg.price;
        published_count++;
    }

    std::cout << "[PUBLISHER] Thread stopping. Published "
              << published_count << " updates.\n";
}