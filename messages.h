#pragma once
#include <atomic>
#include <cstdint>
#include "ring_buffer.h"

// --------------------------------------------------------------------
// UPDATE MESSAGE - Published by engine, consumed by publisher thread
// --------------------------------------------------------------------
// Fields ordered largest → smallest to minimise struct padding.
struct UpdateMessage
{
    uint64_t order_id;      // Order identifier
    uint64_t price;         // Price level
    uint64_t timestamp_ns;  // Event timestamp (hardware TSC)
    uint32_t quantity;      // Quantity involved
    char     type;          // 'N'=New  'X'=Cancel  'T'=Trade  'M'=Modify
    char     side;          // 'B'=Buy  'S'=Sell
};

// --------------------------------------------------------------------
// ZERO-COPY PACKET VIEW
// --------------------------------------------------------------------
// In production HFT the NIC streams packets directly into user-space
// via DMA. The engine reads a pointer to the packet and never copies
// the payload.
//
// Benefits vs full copy:
//   - 16 bytes (PacketView) vs 136 bytes (full message copy)
//   - 4 messages fit in a single 64-byte cache line
//   - 8.5× less inter-core interconnect traffic
struct PacketView
{
    const char* payload;   // Direct pointer to mmap/DMA buffer (zero-copy)
    char        msg_type;  // Cached type byte for immediate switch() dispatch
};

// --------------------------------------------------------------------
// GLOBAL QUEUES
// --------------------------------------------------------------------
// updateBuffer: Engine (producer, Core 2) → Publisher (consumer, Core 3)
// inputQueue:   Network (producer, Core 0) → Engine  (consumer, Core 2)
//
// Declared as plain globals (not singleton functions) so the compiler
// can see them as stable addresses and keep the pointer in a register
// across the hot-path loop — same behaviour as the original single-file code.

extern RingBuffer<UpdateMessage, 1048576> updateBuffer;
extern RingBuffer<PacketView,    524288>  inputQueue;

// Lifecycle signals
extern std::atomic<bool> stopPublisher;
extern std::atomic<bool> stopNetworkThread;
extern std::atomic<bool> stopEngine;
extern std::atomic<bool> startNetworkTraffic; // Fired by main after warmup