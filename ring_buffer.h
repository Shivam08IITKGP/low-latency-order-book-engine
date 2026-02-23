#pragma once
#include <atomic>
#include <cstddef>

// --------------------------------------------------------------------
// LOCK-FREE SPSC RING BUFFER
// --------------------------------------------------------------------
// Single-Producer Single-Consumer (SPSC) design:
//   Producer lives on Core 2 (engine writes to updateBuffer)
//   Consumer lives on Core 3 (publisher reads from updateBuffer)
//
// Key optimisations:
//   - Cache-line alignment (alignas(64)) prevents false sharing between
//     producer and consumer variables.
//   - Cached index copies: each side caches the remote index locally,
//     avoiding expensive atomic loads on the fast path.
//   - Power-of-2 size enables bitwise AND instead of costly modulo (%).
//   - pre_fault_memory() touches every page at startup, eliminating
//     page faults in the hot path.
//
// Template parameter Size MUST be a power of 2.

template<typename T, size_t Size>
class RingBuffer
{
    static_assert((Size != 0) && ((Size & (Size - 1)) == 0),
                  "RingBuffer Size must be a power of 2");

private:
    alignas(64) T buffer[Size];

    // Producer-exclusive variables (written on Core 2)
    alignas(64) std::atomic<size_t> write_idx{0};
    size_t cached_read_idx{0};   // Cached remote index, lives in Core 2's L1

    // Consumer-exclusive variables (written on Core 3)
    alignas(64) std::atomic<size_t> read_idx{0};
    size_t cached_write_idx{0};  // Cached remote index, lives in Core 3's L1

    static constexpr size_t MASK = Size - 1;

public:
    // Returns false if the buffer is full (caller should busy-wait with cpu_pause)
    bool push(const T& msg)
    {
        size_t current_write = write_idx.load(std::memory_order_relaxed);
        size_t next_write    = (current_write + 1) & MASK;

        if (next_write == cached_read_idx)
        {
            // Slow path: refresh cached copy to see if consumer has advanced
            cached_read_idx = read_idx.load(std::memory_order_acquire);
            if (next_write == cached_read_idx)
                return false;   // Truly full
        }

        buffer[current_write] = msg;
        write_idx.store(next_write, std::memory_order_release);
        return true;
    }

    // Returns false if the buffer is empty (caller should busy-wait with cpu_pause)
    bool pop(T& msg)
    {
        size_t current_read = read_idx.load(std::memory_order_relaxed);

        if (current_read == cached_write_idx)
        {
            // Slow path: refresh cached copy to see if producer has advanced
            cached_write_idx = write_idx.load(std::memory_order_acquire);
            if (current_read == cached_write_idx)
                return false;   // Truly empty
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

    // Touch one element per 4 KB page to pre-allocate physical memory.
    // Call this once at startup to eliminate page faults during the hot path.
    void pre_fault_memory()
    {
        const size_t page_size = 4096;
        for (size_t i = 0; i < Size; i += page_size / sizeof(T))
            buffer[i] = T();
    }
};