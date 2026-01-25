#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <map> // We use std::map for now (Red-Black Tree) to keep orders sorted
// Search, Insert, Delete -> all O(logN), in heap, the searching is O(N)
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include "common.h"
/*
 * @brief Low-latency Order Book using pre-allocated memory.
 * DESIGN CHOICE:
 * Uses a flat std::vector<OrderInfo> (size 1M) for O(1) order lookups via ID.
 * Compared to std::unordered_map, this:
 * 1. Eliminates hash collisions and rehashes.
 * 2. Provides better cache locality.
 * 3. Removes heap allocations on the hot path (P99 < 200ns).
 */
class OrderBook {
    // 1. The "View" (For printing Top 10)
    // Keeps track of Total Quantity at each Price
    std::map<uint64_t, uint32_t, std::greater<uint64_t>> bids; // High to Low
    std::map<uint64_t, uint32_t> asks; // Low to High

    // 2. The "Database" (For finding Orders by ID)
    // We need to remember the Price and Side of every Order ID
    struct OrderInfo {
        uint64_t price;
        uint32_t quantity;
        char side; // 'B' or 'S'
    };
    // We use a vector for O(1) direct indexing by Order ID.
    // This is significantly faster than hashing for constant-time lookups.
    std::vector<OrderInfo> order_lookup;

public:
    OrderBook()
    {
        // Reserve space for 1 Million orders upfront.
        // This is 1 allocation at startup, instead of 1,000,000 allocations during runtime.
        order_lookup.resize(1000001);
    }

    // --- HANDLING NEW ORDERS ('N') ---
    void addOrder(uint64_t id, char side, uint64_t price, uint32_t quantity) 
    {
        // A. Store in Lookup (So we can find it later)
        order_lookup[id] = {price, quantity, side};

        // B. Update the Price Level (The View)
        if (side == 'B') bids[price] += quantity;
        else             asks[price] += quantity;
    }

    // --- HANDLING CANCELLATIONS ('X') ---
    void cancelOrder(uint64_t id) 
    {
        // Since we use a vector, lookup is just index access (O(1)).
        if (id >= order_lookup.size()) return;

        // 2. Get details
        OrderInfo& info = order_lookup[id];

        if(info.quantity == 0) return;

        // 3. Remove quantity from the Price Level
        if (info.side == 'B') 
        {
            bids[info.price] -= info.quantity;
            if (bids[info.price] == 0) bids.erase(info.price); // Clean up empty levels
        } 
        else 
        {
            asks[info.price] -= info.quantity;
            if (asks[info.price] == 0) asks.erase(info.price);
        }

        // 4. Forget the order
        info.quantity = 0;
    }

    // --- HANDLING TRADES ('T') ---
    // A trade means TWO orders matched (Buy ID and Sell ID). 
    // We must reduce the quantity for BOTH.
    void executeTrade(uint64_t buy_id, uint64_t sell_id, uint32_t qty) 
    {
        if(order_lookup[buy_id].quantity > 0 && order_lookup[sell_id].quantity > 0)
        {
            // Reduce/Remove the Buy Order
            OrderInfo& buy_info = order_lookup[buy_id];
            
            // Decrease quantity in the Book
            bids[buy_info.price] -= qty;
            if (bids[buy_info.price] == 0) bids.erase(buy_info.price);
            
            // Decrease quantity in the Order
            buy_info.quantity -= qty;
            if (buy_info.quantity == 0) buy_info.quantity = 0; // Filled completely        

            // Reduce/Remove the Sell Order
            OrderInfo& sell_info = order_lookup[sell_id];
            
            // Decrease quantity in the Book
            asks[sell_info.price] -= qty;
            if (asks[sell_info.price] == 0) asks.erase(sell_info.price);

            // Decrease quantity in the Order
            sell_info.quantity -= qty;
            if (sell_info.quantity == 0) sell_info.quantity = 0;
        }
    }

    void printTopOfBook() 
    {
        std::cout << "   [BOOK] ";
        if (bids.empty()) std::cout << "Bids: -";
        else std::cout << "Bid: " << bids.begin()->second << " @ " << bids.begin()->first;
        
        std::cout << " | ";
        
        if (asks.empty()) std::cout << "Asks: -";
        else std::cout << "Ask: " << asks.begin()->second << " @ " << asks.begin()->first;
        
        std::cout << "\n";
    }

    size_t getBidsSize() const { return bids.size(); }
    size_t getAsksSize() const { return asks.size(); }
};

OrderBook orderBook;

int main() {
    const char* filepath = "market_data.bin";

    // 1. Open File
    int fd = open(filepath, O_RDONLY);
    if (fd == -1) { perror("open"); return 1; }

    // 2. Get File Size (for mapping)
    struct stat sb;
    if (fstat(fd, &sb) == -1) { perror("fstat"); return 1; }

    // 3. Memory Map (The "Zero Copy" magic)
    // PROT_READ: We only read
    // MAP_PRIVATE: Changes don't write back to disk (Cow)
    char* file_memory = static_cast<char*>(mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (file_memory == MAP_FAILED) { perror("mmap"); return 1; }

    // 4. Processing Loop
    size_t offset = 0;
    uint32_t messageCount = 0;
    std::vector<long long> latencies;

    auto totalStart = std::chrono::high_resolution_clock::now();

    while (offset < sb.st_size) {
        auto msgStart = std::chrono::high_resolution_clock::now();
        // Read Header
        StreamHeader* header = reinterpret_cast<StreamHeader*>(file_memory + offset);

        // Peek at the type byte (first byte of payload)
        char msgType = file_memory[offset + sizeof(StreamHeader)];

        char* payload = file_memory + offset + sizeof(StreamHeader);

        switch (msgType)
        {
            case 'T':
            {
                TradeMessage* t = reinterpret_cast<TradeMessage*>(payload);
                orderBook.executeTrade(t->buy_order_id, t->sell_order_id, t->quantity);
                break;
            }
            case 'N':
            {
                OrderMessage* o = reinterpret_cast<OrderMessage*>(payload);
                orderBook.addOrder(o->order_id, o->side, o->price, o->quantity);
                break;
            }
            case 'X':
            {
                OrderMessage* o = reinterpret_cast<OrderMessage*>(payload);
                orderBook.cancelOrder(o->order_id);
                break;
            }
            case 'M':
            {
                OrderMessage* o = reinterpret_cast<OrderMessage*>(payload);
                orderBook.cancelOrder(o->order_id);
                orderBook.addOrder(o->order_id, o->side, o->price, o->quantity);
                break;
            }
        }

        // Jump to next message
        offset += header->msg_len + sizeof(StreamHeader);

        auto msgEnd = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(msgEnd - msgStart).count());
        messageCount++;
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(totalEnd - totalStart).count();

    if (messageCount > 0) {
        long long sum = std::accumulate(latencies.begin(), latencies.end(), 0LL);
        double avg = static_cast<double>(sum) / messageCount;
        
        std::sort(latencies.begin(), latencies.end());
        long long p50 = latencies[messageCount / 2];
        long long p99 = latencies[static_cast<size_t>(messageCount * 0.99)];

        std::cout << "\n---------------- Performance Summary ----------------\n";
        std::cout << "Processed messages: " << messageCount << "\n";
        std::cout << "Total time: " << totalDuration << " us\n";
        std::cout << "Average Latency: " << avg << " ns\n";
        std::cout << "P50 Latency: " << p50 << " ns\n";
        std::cout << "P99 Latency: " << p99 << " ns\n";
        std::cout << "-----------------------------------------------------\n";
    }
    std::cout << "Final Book Size -> Bids: " << orderBook.getBidsSize() 
              << " | Asks: " << orderBook.getAsksSize() << "\n";
    // 5. Cleanup
    munmap(file_memory, sb.st_size);
    close(fd);
    return 0;
}