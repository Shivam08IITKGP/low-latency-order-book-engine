#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>

/**
 * CONFIGURATION CONSTANTS
 */
static constexpr uint64_t MAX_PRICE = 100000;
static constexpr size_t MAX_ORDERS = 10000001;

enum class OrderType : uint8_t
{
    GTC = 0,  // Good-Till-Cancelled
    IOC = 1,  // Immediate-or-Cancel
    FOK = 2,  // Fill-or-Kill (Placeholder)
};

class OrderBook
{
public:
    OrderBook();

    /**
     * CORE MATCHING LOGIC (HOT PATH)
     * Time Complexity: O(1)
     */
    void addOrder(uint64_t id,
                  char     side,
                  uint64_t price,
                  uint32_t quantity,
                  uint32_t participant_id = 0,
                  OrderType type = OrderType::GTC);

    void cancelOrder(uint64_t id);

    /**
     * MEMORY MANAGEMENT
     * Forces physical page allocation at startup.
     */
    void pre_fault_memory();

    /**
     * ANALYTICS
     */
    void     printTopOfBook()       const;
    uint64_t getTotalTradedVolume() const { return total_traded_volume; }
    double   getVWAP() const {
        return (total_traded_volume > 0) ? (static_cast<double>(traded_notional_value) / total_traded_volume) : 0.0;
    }

    bool saveSnapshot(const std::string& filename) const;
    bool loadSnapshot(const std::string& filename);

private:
    static constexpr uint32_t NULL_ORDER = std::numeric_limits<uint32_t>::max();

    struct alignas(16) PriceLevel
    {
        uint32_t head      = NULL_ORDER;
        uint32_t tail      = NULL_ORDER;
        uint32_t total_qty = 0;
        uint32_t _pad      = 0;
    };

    // Price-indexed levels
    std::array<PriceLevel, MAX_PRICE + 1> bid_levels{};
    std::array<PriceLevel, MAX_PRICE + 1> ask_levels{};

    uint64_t max_bid_price = 0;
    uint64_t min_ask_price = MAX_PRICE + 1;

    // Occupancy Bitmaps (1 bit per price level)
    static constexpr size_t BITMAP_SIZE = (MAX_PRICE + 64) / 64;
    std::array<uint64_t, BITMAP_SIZE> bid_bitmap{};
    std::array<uint64_t, BITMAP_SIZE> ask_bitmap{};

    inline void set_bid_level  (uint64_t price);
    inline void clear_bid_level(uint64_t price);
    inline void set_ask_level  (uint64_t price);
    inline void clear_ask_level(uint64_t price);

    uint64_t find_next_ask(uint64_t start_price);
    uint64_t find_next_bid(uint64_t start_price);

    struct alignas(16) OrderInfo
    {
        uint32_t price          = 0;
        uint32_t quantity       = 0;
        uint32_t next_id        = NULL_ORDER;
        uint32_t participant_id = 0;
        uint8_t  side           = 0; // 0=Buy, 1=Sell
        uint8_t  _pad[3]        = {};
    };

    // ID-indexed order pool
    std::array<OrderInfo, MAX_ORDERS> order_lookup{};

    uint64_t total_traded_volume   = 0;
    uint64_t traded_notional_value = 0;
};
