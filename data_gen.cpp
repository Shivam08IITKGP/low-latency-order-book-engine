#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "common.h"

/**
 * SYNTHETIC MARKET DATA GENERATOR
 * 
 * Generates a binary stream of StreamHeader + OrderMessage packets
 * to simulate exchange traffic.
 */

int main(int argc, char* argv[])
{
    uint32_t count = 1000000;
    if (argc > 1) {
        try {
            count = std::stoul(argv[1]);
        } catch (...) {
            std::cerr << "Invalid count, defaulting to 1M\n";
        }
    }

    std::ofstream outfile("market_data.bin", std::ios::binary);
    if (!outfile) {
        std::cerr << "Fatal: Could not open market_data.bin\n";
        return 1;
    }

    std::cout << "Generating " << count << " messages...\n";

    for (uint32_t i = 0; i < count; i++) {
        StreamHeader h = {i, static_cast<uint16_t>(sizeof(OrderMessage)), 'N', 0};
        
        char side = (i % 2 == 0) ? 'B' : 'S';
        
        // Base prices
        uint64_t price = (side == 'B') ? 5000 : 5010;
        
        // Periodic matching: Cross the spread every 10 messages
        if (i % 10 == 0) {
            price = (side == 'B') ? 5010 : 4990;
        }

        OrderMessage o = {static_cast<uint64_t>(i + 1), price, 100, 'N', side, {0, 0}};
        outfile.write(reinterpret_cast<char*>(&h), sizeof(h));
        outfile.write(reinterpret_cast<char*>(&o), sizeof(o));
    }

    outfile.close();
    std::cout << "Successfully generated market_data.bin (" << count << " messages).\n";
    return 0;
}