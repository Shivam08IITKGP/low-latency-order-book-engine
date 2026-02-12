#include <iostream>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>
#include <thread>
#include "common_net.h"

int main() {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(MCAST_GRP);
    addr.sin_port = htons(MCAST_PORT);

    std::cout << "EXCHANGE LIVE. Broadcasting to " << MCAST_GRP << ":" << MCAST_PORT << "...\n";

    uint64_t order_id = 1;
    while (true) {
        OrderMessage msg;
        
        // 1. Timestamp the packet (The moment it leaves the exchange)
        msg.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        msg.order_id = order_id++;
        msg.price = 100 + (rand() % 50); // Random price 100-150
        msg.quantity = 1 + (rand() % 100);
        msg.side = (rand() % 2 == 0) ? 'B' : 'S';
        msg.type = 'N';

        // 2. Send the raw bytes
        int nbytes = sendto(sock, &msg, sizeof(msg), 0, (struct sockaddr*)&addr, sizeof(addr));
        if (nbytes < 0) { perror("sendto"); return 1; }

        // Throttle slightly (100k messages/sec) so we don't melt your laptop
        // In real HFT, there is no sleep.
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    close(sock);
    return 0;
}