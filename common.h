#ifndef COMMON_H
#define COMMON_H

#include <cstdint>

// 1. The Stream Header (Wraps every message)
struct StreamHeader {
    uint16_t msg_len;  // Length of the payload following this header
    uint32_t seq_no;   // Sequence number
    char     msg_type; // 'N', 'X', etc.
    char     reserved; 
} __attribute__((packed));

// 2. The Message Body (The "STREAM_DATA")
struct OrderMessage {
    uint64_t order_id;
    uint64_t price;   // Price in ticks
    uint32_t quantity;
    char type;        // 'N' (New), 'M' (Modify), 'X' (Cancel)
    char side;        // 'B' (Buy) or 'S' (Sell)
    char _pad[2];
};

struct TradeMessage {
    char type;        // 'T' (Trade)
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    uint32_t quantity;
    uint64_t price;
} __attribute__((packed));

#endif