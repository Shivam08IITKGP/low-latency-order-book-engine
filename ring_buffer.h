#pragma once
#include <atomic>
#include <cstddef>
#include "cpu_utils.h"

/**
 * LOCK-FREE SINGLE-PRODUCER SINGLE-CONSUMER (SPSC) RING BUFFER
 * 
 * Optimized for low-latency inter-core communication.
 * Features:
 * 1. Cache-line alignment (64 bytes) to prevent false sharing.
 * 2. Local index caching to minimize atomic acquire/release loads on the hot path.
 * 3. Power-of-2 capacity for bitwise wrap-around (Size - 1).
 */

template<typename T, size_t Size>
class RingBuffer
{
    static_assert((Size != 0) && ((Size & (Size - 1)) == 0), "Size must be a power of 2");

private:
    alignas(64) T buffer[Size];
    
    // 1 array
    // 2 pointers (read and write)
    // 2 caches, for cross-core communication

    // Owned by producer (core 0 in our implementation)
    // using alignas(64) to avoid false sharing, and keep these variables
    // in single cache line
    alignas(64) std::atomic<size_t> write_idx{0}; // where the producer will write next
    size_t cached_read_idx{0};
    
    // Owned by consumer (core 2 in our implementation)
    alignas(64) std::atomic<size_t> read_idx{0}; // where the consumer will read next
    size_t cached_write_idx{0};
    
    static constexpr size_t MASK = Size - 1;

public:
    bool push(const T& msg)
    {
        size_t current_write = write_idx.load(std::memory_order_relaxed);
        size_t next_write = (current_write + 1) & MASK;

        // Slow path: Refresh cached read index from consumer core

        // Basically means the write index is catching with read index
        // hence the buffer is full, as we do not want to overwrite unread
        // data

        // this is old cache_read_idx, so we load the 
        // new read index from consumer core and then check again
        // if buffer is full or not
        if (next_write == cached_read_idx)
        {
            // Expensive operation: load with memory order acquire
            cached_read_idx = read_idx.load(std::memory_order_acquire);
            if (next_write == cached_read_idx) return false;
        }

        buffer[current_write] = msg;
        write_idx.store(next_write, std::memory_order_release);
        // release because write to buffer happens before
        // this write_idx store
        return true;
    }

    bool pop(T& msg)
    {
        size_t current_read = read_idx.load(std::memory_order_relaxed);

        // Slow path: Refresh cached write index from producer core
        if (current_read == cached_write_idx)
        {
            cached_write_idx = write_idx.load(std::memory_order_acquire);
            if (current_read == cached_write_idx) return false; // Buffer empty
        }

        msg = buffer[current_read];
        read_idx.store((current_read + 1) & MASK, std::memory_order_release);
        return true;
    }

    void pre_fault_memory()
    // Maps the physical frames to the virtual pages
    {
        const size_t page_size = 4096;
        for (size_t i = 0; i < Size; i += page_size / sizeof(T))
        {
            // static_cast void just means reading the variable, nothing else
            static_cast<void>(buffer[i]);
        }
    }
};