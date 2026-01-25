# Low-Latency Order Book Engine

A high-performance, C++ based limit order book (LOB) engine designed for low-latency trading systems. This project demonstrates core systems programming concepts essential for HFT, including memory-mapped files (zero-copy), efficient data structures, and nanosecond-level performance measurement.

## Key Features

- **Zero-Copy Processing**: Uses `mmap` to map market data files directly into memory, avoiding expensive context switches and buffer copies.
- **Efficient Book Management**: Implements a `map`-based price level tracking for high-to-low (Bids) and low-to-high (Asks) sorting, combined with `unordered_map` for O(1) order lookups and cancellations.
- **Nanosecond Latency Measurement**: Integrated `std::chrono` timing to track per-message processing latency and total throughput.
- **HFT Statistics**: Provides P50 (median) and P99 latency reports to identify tail latency issues.

## Architecture

The engine processes a binary stream of market data. Each message consists of a `StreamHeader` followed by a payload (`OrderMessage` or `TradeMessage`).

### Design Decisions

- **Uint64 Price Representation**: Prices are stored as `uint64_t` (ticks/cents) to eliminate floating-point inaccuracies.
- **Memory Mapping**: Leveraging OS-level page cache for extremely fast file I/O.
- **Packed Structs**: Using `__attribute__((packed))` to ensure memory layout matches the binary wire format.

## Performance Summary

Example output with 100,000 messages:

```text
---------------- Performance Summary ----------------
Processed messages: 100000
Total time: 24508 us
Average Latency: 157 ns
P50 Latency: 89 ns
P99 Latency: 188 ns
-----------------------------------------------------
```

_(Note: Latency varies based on hardware and CPU pinning)_

## Getting Started

### Prerequisites

- GCC (supporting C++17 or later)
- Make

### Build and Run

```bash
# Build the generator and engine
make all

# Generate test data (100k messages)
./data_gen

# Run the engine
./engine
```

## Professional Aspirations

This project is built as part of my journey toward a **Systems Engineer role in High-Frequency Trading (HFT)**. It reflects my focus on hardware-software empathy, low-latency optimization, and robust systems design.
