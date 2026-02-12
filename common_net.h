#pragma once
#include <cstdint>

// Multicast Group IP and Port (Local Simulation)
#define MCAST_GRP "239.0.0.1"
#define MCAST_PORT 12345

#pragma pack(push, 1) // Force compiler to pack structs (No padding bytes)

struct OrderMessage {
    uint64_t timestamp; // Time the packet left the exchange (ns)
    uint64_t order_id;
    uint64_t price;
    uint32_t quantity;
    char side;          // 'B' or 'S'
    char type;          // 'N' (New), 'X' (Cancel)
};

#pragma pack(pop)