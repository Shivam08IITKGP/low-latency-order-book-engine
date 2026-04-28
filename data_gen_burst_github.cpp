#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <cstring>
#include "common.h"

int main() {
    const uint32_t num_adds = 750000;
    const uint32_t num_cancels = 100000;

    std::ofstream outfile("market_data.bin", std::ios::binary);
    if (!outfile) return 1;

    std::cout << "Generating " << num_adds << " Add orders (GitHub Layout)...\n";
    for (uint32_t i = 0; i < num_adds; i++) {
        StreamHeader h;
        h.msg_len = sizeof(OrderMessage);
        h.seq_no = i;
        h.reserved = 0;

        OrderMessage o;
        o.type = 'N';
        o.order_id = i + 1;
        o.side = 'B';
        o.quantity = 10;
        o.price = 5000 + (i % 100);

        outfile.write(reinterpret_cast<char*>(&h), sizeof(h));
        outfile.write(reinterpret_cast<char*>(&o), sizeof(o));
    }

    std::cout << "Generating " << num_cancels << " Randomized Cancel messages...\n";
    std::vector<uint32_t> ids(num_adds);
    for (uint32_t i = 0; i < num_adds; i++) ids[i] = i + 1;
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(ids.begin(), ids.end(), g);

    for (uint32_t i = 0; i < num_cancels; i++) {
        StreamHeader h;
        h.msg_len = sizeof(OrderMessage);
        h.seq_no = num_adds + i;
        h.reserved = 0;

        OrderMessage o;
        o.type = 'X';
        o.order_id = ids[i];
        o.side = 'B';
        o.quantity = 0;
        o.price = 0;

        outfile.write(reinterpret_cast<char*>(&h), sizeof(h));
        outfile.write(reinterpret_cast<char*>(&o), sizeof(o));
    }

    outfile.close();
    std::cout << "Successfully generated market_data.bin (" << (num_adds + num_cancels) << " messages).\n";
    return 0;
}
