#include "cpu_utils.h"
#include <iostream>
#include <fstream>
#include <string>
#include <pthread.h>
#include <sched.h>

bool isCoreIsolated(int core_id)
{
    std::ifstream isolcpus("/sys/devices/system/cpu/isolated");
    if (!isolcpus.is_open())
        return false;

    std::string isolated;
    std::getline(isolcpus, isolated);

    // Parsed format may be "2", "2-3", or "2,4"
    return isolated.find(std::to_string(core_id)) != std::string::npos;
}

bool pinThreadToCore(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (result != 0)
    {
        std::cerr << "[ERROR] Failed to pin thread to core " << core_id << "\n";
        return false;
    }

    std::cout << "[CPU] Thread pinned to core " << core_id;

    if (!isCoreIsolated(core_id))
        std::cout << " [WARNING: NOT ISOLATED]";

    std::cout << "\n";
    return true;
}