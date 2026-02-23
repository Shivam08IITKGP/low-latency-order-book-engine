#include "messages.h"

// Global queue definitions — plain globals so the compiler treats their
// addresses as link-time constants and can keep them in registers.
RingBuffer<UpdateMessage, 1048576> updateBuffer;
RingBuffer<PacketView,    524288>  inputQueue;

std::atomic<bool> stopPublisher{false};
std::atomic<bool> stopNetworkThread{false};
std::atomic<bool> stopEngine{false};
std::atomic<bool> startNetworkTraffic{false};