#include <iostream>
#include <vector>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>
#include <fcntl.h>
#include "common_net.h"

// --- COPY YOUR FastOrderBook CLASS HERE ---
// (For brevity, I assume FastOrderBook code is pasted here or included)
// Just create a dummy one for the network test if you want to keep code short.
class FastOrderBook
{
public:
    void addOrder(uint64_t id, char side, uint64_t price, uint32_t qty)
    {
        // Your logic...
    }
};

void pinToCore(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

    if (result != 0)
    {
        std::cerr << "Error pinning to core " << core_id << "\n";
    }
    else
    {
        std::cout << "PROCESS PINNED TO CORE " << core_id << " (NO MIGRATION)\n";
    }
}

int main()
{
    pinToCore(3);
    FastOrderBook book;

    // 1. Create UDP Socket
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    fcntl(sock, F_SETFL, O_NONBLOCK);
    if (sock < 0)
    {
        perror("socket");
        return 1;
    }

    // 2. Allow reusing the port (so you can restart quickly)
    int reuse = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char *)&reuse, sizeof(reuse)) < 0)
    {
        perror("setting SO_REUSEADDR");
        return 1;
    }

    // 3. Bind to the Port
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY); // Listen on all interfaces
    addr.sin_port = htons(MCAST_PORT);

    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        perror("bind");
        return 1;
    }

    // 4. Join Multicast Group
    struct ip_mreq mreq;
    mreq.imr_multiaddr.s_addr = inet_addr(MCAST_GRP);
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *)&mreq, sizeof(mreq)) < 0)
    {
        perror("setsockopt");
        return 1;
    }

    std::cout << "HFT ENGINE LIVE. Listening on " << MCAST_GRP << ":" << MCAST_PORT << "...\n";

    char buffer[1024];
    uint64_t total_latency = 0;
    uint64_t count = 0;

    // 5. The Hot Loop
    while (true)
    {
        struct sockaddr_in src_addr;
        socklen_t addrlen = sizeof(src_addr);

        // RECV: This blocks until a packet arrives
        // In real HFT, we use "busy spinning" (recvmmsg with MSG_DONTWAIT)

        int nbytes = recvfrom(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&src_addr, &addrlen);

        if (nbytes < 0)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                continue;
            else
            {
                perror("recvfrom");
                break;
            }
        }

        // Capture Receive Time immediately
        uint64_t now = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        // Parse
        if (nbytes == sizeof(OrderMessage))
        {
            OrderMessage *msg = reinterpret_cast<OrderMessage *>(buffer);

            // Calculate "Wire Latency" (Time in flight + OS overhead)
            uint64_t latency = now - msg->timestamp;
            total_latency += latency;
            count++;

            // Execute
            book.addOrder(msg->order_id, msg->side, msg->price, msg->quantity);

            // Report every 10,000 packets
            if (count % 10000 == 0)
            {
                std::cout << "Processed 10k orders. Avg Network Latency: " << (total_latency / count) << " ns\n";
                total_latency = 0;
                count = 0;
            }
        }
    }
    close(sock);
    return 0;
}