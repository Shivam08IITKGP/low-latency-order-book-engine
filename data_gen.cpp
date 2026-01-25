#include <iostream>
#include <fstream>
#include "common.h"

int main() {
    std::ofstream outfile("market_data.bin", std::ios::binary);

    // --- Message 1: New Buy Order ---
    for (int i = 0; i < 100000; i++) {
        // Generate a New Order
        StreamHeader h = {sizeof(OrderMessage), (uint32_t)i, 0};
        OrderMessage o = {'N', (uint64_t)i, (i % 2 == 0 ? 'B' : 'S'), 100, (uint64_t)(5000 + (i % 10))};
        
        outfile.write(reinterpret_cast<char*>(&h), sizeof(h));
        outfile.write(reinterpret_cast<char*>(&o), sizeof(o));
    }

    // --- Message 2: Trade Execution ---
    StreamHeader h2 = {sizeof(TradeMessage), 2, 0};
    TradeMessage t1 = {'T', 101, 205, 50, 5000}; // Trade 50 units @ 50.00

    outfile.write(reinterpret_cast<char*>(&h2), sizeof(h2));
    outfile.write(reinterpret_cast<char*>(&t1), sizeof(t1));

    outfile.close();
    std::cout << "Generated market_data.bin successfully.\n";
    return 0;
}