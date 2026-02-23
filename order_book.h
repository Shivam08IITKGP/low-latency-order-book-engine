#pragma once
#include <array>
#include <cstdint>
#include <cstddef>

static constexpr uint64_t MAX_PRICE = 100000;

// --------------------------------------------------------------------
// ORDER BOOK
// --------------------------------------------------------------------
// Data structures:
//   bids[price] / asks[price]  — aggregated quantity at each price level
//                                (flat array indexed by price → O(1) access)
//   order_lookup[order_id]     — per-order info stored as a compact 8-byte
//                                bitfield (vs 16 bytes unpadded), halving the
//                                memory footprint and improving cache density
//   bid_bitmap / ask_bitmap    — one bit per price level; enables O(1) best-
//                                bid/ask search via __builtin_ctzll /
//                                __builtin_clzll instead of a linear scan
//
// Hot-path properties:
//   - No heap allocation (all storage is std::array, allocated at program start)
//   - O(1) order lookup (direct index, no hash collisions, no rehashing)
//   - Excellent cache locality (contiguous arrays)

class OrderBook
{
public:
    OrderBook();

    // Core operations — called from the engine hot path
    void addOrder    (uint64_t id, char side, uint64_t price, uint32_t quantity);
    void cancelOrder (uint64_t id);
    void executeTrade(uint64_t buy_id, uint64_t sell_id, uint32_t qty);
    void modifyOrder (uint64_t id, char new_side, uint64_t new_price, uint32_t new_quantity);
    void executeMarketOrder(char side, uint32_t quantity);

    // Startup: touch every page so the OS allocates physical memory upfront
    void pre_fault_memory();

    // Diagnostics
    void printTopOfBook() const;
    uint64_t getTotalTradedVolume() const { return total_traded_volume; }
    size_t   getBidsSize()          const { return bids.size(); }
    size_t   getAsksSize()          const { return asks.size(); }

private:
    // ------------------------------------------------------------
    // Price-level aggregation
    // ------------------------------------------------------------
    std::array<uint32_t, MAX_PRICE + 1> bids{};
    std::array<uint32_t, MAX_PRICE + 1> asks{};

    uint64_t max_bid_price = 0;
    uint64_t min_ask_price = MAX_PRICE + 1;

    // Bitmaps: one bit per price level for O(1) best-price search
    static constexpr size_t BITMAP_SIZE = (MAX_PRICE + 64) / 64;
    std::array<uint64_t, BITMAP_SIZE> bid_bitmap{};
    std::array<uint64_t, BITMAP_SIZE> ask_bitmap{};

    // Bitmap helpers (inline for zero call overhead)
    inline void set_bid_level  (uint64_t price);
    inline void clear_bid_level(uint64_t price);
    inline void set_ask_level  (uint64_t price);
    inline void clear_ask_level(uint64_t price);

    // Best-price search using bit scanning intrinsics
    uint64_t find_next_ask(uint64_t start_price);
    uint64_t find_next_bid(uint64_t start_price);

    // ------------------------------------------------------------
    // Order lookup table
    // ------------------------------------------------------------
    // COMPACT BITFIELD: 8 bytes per order (vs 16 bytes naive layout)
    //   price    : 32 bits — covers MAX_PRICE = 100,000
    //   quantity : 31 bits — up to ~2 billion shares
    //   side     : 1  bit  — 0='B'(Buy), 1='S'(Sell)
    //
    // Savings: 8 MB vs 16 MB for 1 M orders → 2× better L3 cache fit,
    //          half the TLB pressure.
    struct alignas(8) OrderInfo
    {
        uint64_t price    : 32;
        uint64_t quantity : 31;
        uint64_t side     :  1; // 0='B', 1='S'
    };

    std::array<OrderInfo, 1000001> order_lookup{};

    uint64_t total_traded_volume = 0;
};