#pragma once
#include <cstddef>

// --------------------------------------------------------------------
// NETWORK THREAD (Core 0) — zero-copy packet producer
// --------------------------------------------------------------------
// Simulates a NIC streaming packets into user-space via DMA.
// Produces lightweight PacketView objects (pointer + type byte) instead
// of copying full message payloads.  The engine thread then casts the
// pointer directly to the appropriate wire-format struct.
//
// Production path:
//   NIC → DMA → user-space ring buffer → PacketView → Engine

void networkThread(char* file_memory, size_t file_size);

// --------------------------------------------------------------------
// PUBLISHER THREAD (Core 3) — ring buffer consumer
// --------------------------------------------------------------------
// Drains UpdateMessage entries from the update ring buffer.
// In production this would fan out to:
//   - Market data feeds
//   - Client connections
//   - Logging / persistence layer
void publisherThread();