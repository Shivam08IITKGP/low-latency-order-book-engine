#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <map>              // Previously considered for price levels (RB-tree, O(logN))
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include "common.h"

/*
 * Low-Latency Order Book Implementation
 *
 * DESIGN DECISIONS:
 * - Price levels stored in flat vectors indexed by price.
 * - Orders stored in a pre-allocated vector indexed by order ID.
 *
 * Benefits:
 * 1. O(1) order lookup (direct indexing).
 * 2. No hash collisions (vs unordered_map).
 * 3. No rehashing.
 * 4. Excellent cache locality.
 * 5. No heap allocations in hot path.
 */

const uint64_t MAX_PRICE = 100000;

class OrderBook
{
    // --------------------------------------------------------------------
    // 1. PRICE LEVEL VIEW (Aggregated quantity per price)
    // --------------------------------------------------------------------
    // bids[price] = total bid quantity at that price
    // asks[price] = total ask quantity at that price
    std::vector<uint32_t> bids;
    std::vector<uint32_t> asks;

    // Track current best bid and best ask
    uint64_t max_bid_price = 0;
    uint64_t min_ask_price = MAX_PRICE + 1;

    // --------------------------------------------------------------------
    // 2. ORDER LOOKUP TABLE (OrderID -> OrderInfo)
    // --------------------------------------------------------------------
    struct OrderInfo
    {
        uint64_t price;
        uint32_t quantity;
        char side; // 'B' = Buy, 'S' = Sell
    };

    // Direct indexing by Order ID (O(1))
    std::vector<OrderInfo> order_lookup;

    // Total traded volume accumulator
    uint64_t total_traded_volume;

public:
    OrderBook()
    {
        // Pre-allocate space for up to 1M orders
        order_lookup.resize(1000001);

        // Allocate full price range
        bids.resize(MAX_PRICE + 1);
        asks.resize(MAX_PRICE + 1);

        total_traded_volume = 0;
    }

    // --------------------------------------------------------------------
    // NEW ORDER ('N')
    // --------------------------------------------------------------------
    void addOrder(uint64_t id, char side, uint64_t price, uint32_t quantity)
    {
        // Store order details for future lookup
        order_lookup[id] = {price, quantity, side};

        // Update aggregated price level
        if (side == 'B')
        {
            bids[price] += quantity;
            max_bid_price = std::max(price, max_bid_price);
        }
        else
        {
            asks[price] += quantity;
            min_ask_price = std::min(price, min_ask_price);
        }
    }

    // --------------------------------------------------------------------
    // CANCEL ORDER ('X')
    // --------------------------------------------------------------------
    void cancelOrder(uint64_t id)
    {
        // Validate ID range
        if (id >= order_lookup.size())
            return;

        OrderInfo &info = order_lookup[id];

        // Ignore if already removed
        if (info.quantity == 0)
            return;

        // Remove quantity from price level
        if (info.side == 'B')
        {
            bids[info.price] -= info.quantity;

            // Recompute best bid if needed
            if (info.price == max_bid_price && bids[info.price] == 0)
                while (max_bid_price > 0 && bids[max_bid_price] == 0)
                    max_bid_price--;
        }
        else
        {
            asks[info.price] -= info.quantity;

            // Recompute best ask if needed
            if (info.price == min_ask_price && asks[info.price] == 0)
                while (min_ask_price <= MAX_PRICE && asks[min_ask_price] == 0)
                    min_ask_price++;
        }

        // Mark order as removed
        info.quantity = 0;
    }

    // --------------------------------------------------------------------
    // TRADE ('T')
    // --------------------------------------------------------------------
    // Reduces quantity of both matched orders
    void executeTrade(uint64_t buy_id, uint64_t sell_id, uint32_t qty)
    {
        if (order_lookup[buy_id].quantity > 0 &&
            order_lookup[sell_id].quantity > 0)
        {
            // ----- BUY SIDE -----
            OrderInfo &buy_info = order_lookup[buy_id];
            bids[buy_info.price] -= qty;
            buy_info.quantity -= qty;

            if (buy_info.price == max_bid_price &&
                bids[buy_info.price] == 0)
                while (max_bid_price > 0 && bids[max_bid_price] == 0)
                    max_bid_price--;

            // ----- SELL SIDE -----
            OrderInfo &sell_info = order_lookup[sell_id];
            asks[sell_info.price] -= qty;
            sell_info.quantity -= qty;

            if (sell_info.price == min_ask_price &&
                asks[sell_info.price] == 0)
                while (min_ask_price <= MAX_PRICE &&
                       asks[min_ask_price] == 0)
                    min_ask_price++;

            total_traded_volume += qty;
        }
    }

    // Print best bid/ask snapshot
    void printTopOfBook()
    {
        std::cout << "   [BOOK] Bid: "
                  << (max_bid_price > 0 ? bids[max_bid_price] : 0)
                  << " @ " << max_bid_price
                  << " | Ask: "
                  << (min_ask_price <= MAX_PRICE ? asks[min_ask_price] : 0)
                  << " @ " << min_ask_price << "\n";
    }

    uint64_t getTotalTradedVolume() const { return total_traded_volume; }
    size_t getBidsSize() const { return bids.size(); }
    size_t getAsksSize() const { return asks.size(); }
};

OrderBook orderBook;

int main()
{
    const char *filepath = "market_data.bin";

    // --------------------------------------------------------------------
    // 1. Open file
    // --------------------------------------------------------------------
    int fd = open(filepath, O_RDONLY);
    if (fd == -1)
    {
        perror("open");
        return 1;
    }

    // --------------------------------------------------------------------
    // 2. Get file size (required for mmap)
    // --------------------------------------------------------------------
    struct stat sb;
    if (fstat(fd, &sb) == -1)
    {
        perror("fstat");
        return 1;
    }

    // --------------------------------------------------------------------
    // 3. Memory map file (zero-copy read)
    // --------------------------------------------------------------------
    // PROT_READ  : read-only mapping
    // MAP_PRIVATE: copy-on-write (no disk modification)
    char *file_memory =
        static_cast<char *>(mmap(nullptr, sb.st_size,
                                 PROT_READ, MAP_PRIVATE, fd, 0));

    if (file_memory == MAP_FAILED)
    {
        perror("mmap");
        return 1;
    }

    // --------------------------------------------------------------------
    // 4. Processing loop
    // --------------------------------------------------------------------
    size_t offset = 0;
    uint32_t messageCount = 0;
    std::vector<long long> latencies;

    // Warmup phase to prime CPU caches and branch predictors
    std::cout << "Warming up caches...\n";
    for (int i = 0; i < 10000; i++)
    {
        orderBook.addOrder(i, 'B', i % 1000, 10);
        orderBook.cancelOrder(i);
    }
    std::cout << "Warmup complete. Starting benchmark.\n";

    auto totalStart = std::chrono::high_resolution_clock::now();

    while (offset < sb.st_size)
    {
        auto msgStart = std::chrono::high_resolution_clock::now();

        // Parse message header
        StreamHeader *header =
            reinterpret_cast<StreamHeader *>(file_memory + offset);

        // First byte of payload is message type
        char msgType =
            file_memory[offset + sizeof(StreamHeader)];

        char *payload =
            file_memory + offset + sizeof(StreamHeader);

        switch (msgType)
        {
        case 'T':
        {
            TradeMessage *t =
                reinterpret_cast<TradeMessage *>(payload);
            orderBook.executeTrade(
                t->buy_order_id,
                t->sell_order_id,
                t->quantity);
            break;
        }
        case 'N':
        {
            OrderMessage *o =
                reinterpret_cast<OrderMessage *>(payload);
            orderBook.addOrder(
                o->order_id,
                o->side,
                o->price,
                o->quantity);
            break;
        }
        case 'X':
        {
            OrderMessage *o =
                reinterpret_cast<OrderMessage *>(payload);
            orderBook.cancelOrder(o->order_id);
            break;
        }
        case 'M': // Modify = Cancel + Add
        {
            OrderMessage *o =
                reinterpret_cast<OrderMessage *>(payload);
            orderBook.cancelOrder(o->order_id);
            orderBook.addOrder(
                o->order_id,
                o->side,
                o->price,
                o->quantity);
            break;
        }
        }

        // Advance to next message
        offset += header->msg_len + sizeof(StreamHeader);

        auto msgEnd = std::chrono::high_resolution_clock::now();

        latencies.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                msgEnd - msgStart)
                .count());

        messageCount++;
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalDuration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            totalEnd - totalStart)
            .count();

    // --------------------------------------------------------------------
    // Latency Statistics
    // --------------------------------------------------------------------
    if (messageCount > 0)
    {
        long long sum =
            std::accumulate(latencies.begin(),
                            latencies.end(), 0LL);

        double avg =
            static_cast<double>(sum) / messageCount;

        std::sort(latencies.begin(), latencies.end());

        long long p50 =
            latencies[messageCount / 2];

        long long p99 =
            latencies[static_cast<size_t>(
                messageCount * 0.99)];

        std::cout << "\n---------------- Performance Summary ----------------\n";
        std::cout << "Processed messages: " << messageCount << "\n";
        std::cout << "Total time: " << totalDuration << " us\n";
        std::cout << "Average Latency: " << avg << " ns\n";
        std::cout << "P50 Latency: " << p50 << " ns\n";
        std::cout << "P99 Latency: " << p99 << " ns\n";
        std::cout << "-----------------------------------------------------\n";
    }

    std::cout << "Final Book Size -> Bids: "
              << orderBook.getBidsSize()
              << " | Asks: "
              << orderBook.getAsksSize() << "\n";

    std::cout << "Total Traded Volume = "
              << orderBook.getTotalTradedVolume() << "\n";

    // Cleanup memory mapping
    munmap(file_memory, sb.st_size);
    close(fd);

    return 0;
}
