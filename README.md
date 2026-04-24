# Low Latency Order Book Engine

A high-performance C++ matching engine designed for sub-microsecond event processing. This system implements production-grade techniques including zero-copy networking simulation, lock-free inter-thread communication, and hardware-level serialized timing.

## Performance Benchmarks (x86-64, 2.5 GHz)

| Metric | Result |
| :--- | :--- |
| **P50 Latency (Median)** | **21 ns** |
| **Mean Latency** | **34 ns** |
| **P99 Latency** | **43 ns** |
| **Throughput** | **19.2M messages/sec** |

*Note: Benchmarks performed with Core 2 isolated via `isolcpus`.*

## Core Architecture

The engine utilizes a three-thread pipelined architecture to maximize instruction throughput and minimize cache contention.

1.  **Network Thread (Core 0)**: Simulates a zero-copy NIC driver. It maps market data into the process space and builds 16-byte `PacketView` objects (pointer + type) rather than copying payloads.
2.  **Engine Thread (Core 2)**: The hot path. It consumes `PacketViews`, executes the matching logic, and produces updates. It utilizes direct-indexed tables and hardware bitmasks for O(1) performance.
3.  **Publisher Thread (Core 3)**: Handles downstream persistence. It drains the update queue and writes events to a memory-mapped sequential journal, offloading all I/O from the engine.

## Technical Specifications

*   **Price-Time Priority (FIFO)**: Implemented via intrusive singly-linked lists per price level.
*   **O(1) Order Lookup**: Direct-indexed order pool (10M capacity).
*   **O(1) Price Search**: Hardware bitmasks (`__builtin_ctzll`) for instant best-bid/ask identification.
*   **Branchless Dispatch**: `cancelOrder` utilizes a function pointer table and C++ templates to eliminate branches in the cancellation hot path.
*   **Cache Optimization**: `SideLevels` (bid/ask arrays) are `alignas(64)` to ensure no false sharing and optimal cache line utilization.
*   **Zero Heap Allocation**: All memory is pre-allocated and pre-faulted at startup to eliminate demand-paging jitter.
*   **Lock-Free SPSC Queues**: Cache-line aligned Single-Producer Single-Consumer ring buffers for inter-thread communication.
*   **Serialized Timing**: `LFENCE + RDTSC` / `RDTSCP + LFENCE` sandwich for cycle-accurate measurement.

## Build and Execute

### Prerequisites
- GCC 9.0+ or Clang 10.0+ (C++17 support)
- Linux x86-64 environment

### Run Instructions
```bash
# 1. Build the engine and generator
make clean && make -j$(nproc)

# 2. Generate 1M messages for benchmark
./data_gen 1000000

# 3. Execute benchmark
./orderbook
```

## System Optimization (Recommended)

To achieve deterministic sub-microsecond tail latencies, isolate the engine cores in your GRUB configuration:
```bash
# Edit /etc/default/grub
GRUB_CMDLINE_LINUX="isolcpus=0,2,3 nohz_full=0,2,3 rcu_nocbs=0,2,3"
```
Followed by `sudo update-grub` and a system reboot.
