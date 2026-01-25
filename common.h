#ifndef COMMON_H
#define COMMON_H

#include <cstdint>

// 1. The Stream Header (Wraps every message)
struct StreamHeader {
    uint16_t msg_len; // Length of the payload following this header
    uint32_t seq_no;  // Sequence number
    uint16_t unused;  // "One field not useful"
} __attribute__((packed));

// 2. The Message Body (The "STREAM_DATA")
struct OrderMessage {
    char type;        // 'N' (New), 'M' (Modify), 'X' (Cancel)
    uint64_t order_id;
    char side;        // 'B' (Buy) or 'S' (Sell)
    uint32_t quantity;
    uint64_t price;   // Using uint64_t for price (in cents/ticks) is standard to avoid float errors
} __attribute__((packed));

struct TradeMessage {
    char type;        // 'T' (Trade)
    uint64_t buy_order_id;
    uint64_t sell_order_id;
    uint32_t quantity;
    uint64_t price;
} __attribute__((packed));

#endif