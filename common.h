#ifndef COMMON_H
#define COMMON_H

#include <cstdint>

// 1. The Stream Header (Wraps every message)
struct StreamHeader {
    uint32_t seq_no;   // Sequence number - 4 bytes
    uint16_t msg_len;  // Length of the payload - 2 bytes
    char     msg_type; // 'N', 'X', etc. - 1 byte
    char     reserved; // 1 byte
};

// 2. The Message Body (The "STREAM_DATA")
struct OrderMessage {
    uint64_t order_id;// 8 bytes
    uint64_t price;   // Price in ticks - 8 bytes
    uint32_t quantity;// 4 bytes
    char type;        // 'N' (New), 'M' (Modify), 'X' (Cancel) - 1 byte
    char side;        // 'B' (Buy) or 'S' (Sell) - 1 byte
    char _pad[2];     // to make it 24 bytes, multiple of
    // largest primitive in this struct i.e. uint64_t
    // hence preventing any garbage value to leak into
    // network stream
};
/*
Didn't do the attribute packed here, as wanted
to prevent the misalignment of the uint64_t field on
on 8-byte boundaries, it can load the price and id in a 
single clock cycle without hardware penalty
*/

struct TradeMessage {
    uint64_t buy_order_id;  // 8 bytes
    uint64_t sell_order_id; // 8 bytes
    uint64_t price;         // 8 bytes
    uint32_t quantity;      // 4 bytes
    char type;              // 'T' (Trade) - 1 byte
    char _pad[3];           // 3 bytes padding
};

#endif