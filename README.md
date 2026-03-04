# Low-Latency Order Book Engine

A production-grade, multi-threaded C++ matching engine built around zero-copy
networking, lock-free inter-thread communication, hardware-level timing, and CPU
affinity. Designed to reflect real techniques used in latency-sensitive trading
systems.

---

## Performance (5-Run Benchmark, Non-Isolated Core)

Measured with 100,001 messages on a standard Linux desktop -- no kernel CPU
isolation applied.

| Run     | Throughput    | Min   | P50   | P90   | P99   | P99.9 |
|---------|--------------|-------|-------|-------|-------|-------|
| 1       | 14.07 M/s    | 27 ns | 33 ns | 35 ns | 45 ns | 54 ns |
| 2       | 14.43 M/s    | 25 ns | 30 ns | 34 ns | 43 ns | 71 ns |
| 3       | 14.42 M/s    | 25 ns | 33 ns | 35 ns | 44 ns | 57 ns |
| 4       | 14.58 M/s    | 25 ns | 31 ns | 34 ns | 42 ns | 55 ns |
| 5       | 14.11 M/s    | 25 ns | 33 ns | 35 ns | 44 ns | 61 ns |
| **Avg** | **14.32 M/s** | **~25 ns** | **~32 ns** | **~35 ns** | **~44 ns** | **~60 ns** |

**Timing method:** LFENCE+RDTSC (start) / RDTSCP+LFENCE (end) -- Intel-recommended
hardware TSC sandwich, per-message.
**CPU frequency:** ~2499 MHz (calibrated at startup via TSC).

> P99.9 variance is caused by the OS scheduler interrupting the non-isolated
> engine core. With `isolcpus=2 nohz_full=2 rcu_nocbs=2` in the kernel command
> line, P99.9 is expected to drop below 5 us consistently.

---

## Architecture

```
+------------------------------------------------------------------+
|             Order Book Engine -- Thread Pipeline                 |
+------------------------------------------------------------------+
|                                                                  |
|  Network Thread  (Core 0)                                        |
|  ----------------------------------------------------------------|
|  Memory-maps market_data.bin; builds zero-copy PacketViews       |
|  (16-byte pointer + cached type byte, not a 136-byte copy)       |
|      |                                                           |
|      v  Lock-Free SPSC Input Queue  (524 K slots)                |
|                                                                  |
|  Engine Thread  (Core 2)                                         |
|  ----------------------------------------------------------------|
|  Pops PacketViews; casts payload pointer directly                |
|  Dispatches: addOrder / cancelOrder / executeTrade /             |
|              modifyOrder / executeMarketOrder                    |
|  Price-Time (FIFO) priority matching with STP                    |
|  Per-message RDTSC sandwich timing                               |
|      |                                                           |
|      v  Lock-Free SPSC Update Queue  (1 M slots)                 |
|                                                                  |
|  Publisher Thread  (Core 3)                                      |
|  ----------------------------------------------------------------|
|  Drains UpdateMessages                                           |
|  Appends each event to the memory-mapped journal (zero-copy)     |
|  In production: fan-out to market-data feeds, FIX gateways,     |
|  and client connections                                          |
|                                                                  |
|  Persistence Files  (disk, kernel page-cache write-back)         |
|  ----------------------------------------------------------------|
|  event_journal.bin       -- sequential log of all UpdateMessages |
|  orderbook_snapshot.bin  -- full book state snapshot on exit     |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Matching Engine Features

### Price-Time Priority (FIFO)

Each price level maintains a singly-linked FIFO queue of individual order IDs
through `OrderInfo::next_id`. When an incoming order crosses the book, it walks
the opposite side level-by-level, filling the oldest resting order first at
each price. This is the same discipline used by NASDAQ and most other lit venues.

### Order Types

| Type | Behaviour |
|------|-----------|
| **GTC** (Good-Till-Cancelled) | Rest on the book at the back of the queue if not fully filled |
| **IOC** (Immediate-or-Cancel) | Fill as much as possible; cancel any unfilled remainder |
| **FOK** (Fill-or-Kill)        | Fill the entire quantity or cancel the whole order |

### Self-Trade Prevention (STP)

Every order carries a `participant_id`. If an incoming order would match against
a resting order owned by the same participant, the engine cancels the resting
order (Cancel-Resting policy) and continues walking the book. This prevents wash
trading, which is a regulatory violation on all major exchanges.

### Lazy Cancellation

`cancelOrder` sets `quantity = 0` and decrements `PriceLevel::total_qty`
immediately (keeping visible depth correct), but leaves the node in the linked
list. The node is evicted from the head of the queue at O(1) cost the next time
the matching loop visits that level -- no list traversal required.

---

## Persistence Layer

### Memory-Mapped Event Journal (`journal.h`)

Every `UpdateMessage` published by the engine is appended to
`event_journal.bin` on the Publisher thread (Core 3), keeping the Engine
thread entirely free of I/O work.

Appending is a plain memory store -- the kernel's page-cache write-back handles
the physical disk write asynchronously. This avoids the ~1-3 us per-call overhead
of `write()` syscalls entirely.

| Approach | Throughput impact |
|----------|------------------|
| `std::ofstream` / `fprintf` | ~100 K msgs/sec (blocks on disk I/O) |
| `mmap` + `MAP_SHARED`       | < 5% throughput reduction (async page-cache) |

### State Snapshots (`OrderBook::saveSnapshot`)

On clean shutdown the engine serialises the full book state -- price level queues,
order lookup table, bitmaps, and running totals -- into `orderbook_snapshot.bin`.
On restart, `loadSnapshot()` restores this state in milliseconds, allowing the
engine to resume from the last known-good position rather than replaying the
entire journal.

---

## Analytics

### Volume-Weighted Average Price (VWAP)

The matching loop accumulates two integer counters at the cost of two additions
and one multiplication per trade -- no division in the hot path:

```cpp
total_traded_volume   += matched;
traded_notional_value += static_cast<uint64_t>(matched) * execution_price;
```

The division is deferred to the `getVWAP()` getter, which is called only during
reporting. This keeps the per-trade overhead at exactly two integer ALU
instructions, avoiding the ~20-40 cycle penalty of `fdiv` in the critical path.

---

## Module Structure

```
.
+-- common.h          -- Wire-format structs: StreamHeader, OrderMessage,
|                        TradeMessage
+-- timing.h/.cpp     -- Hardware TSC timing utilities
|                          rdtsc_start()           LFENCE+RDTSC (per-message)
|                          rdtsc_end()             RDTSCP+LFENCE (per-message)
|                          rdtsc_start_full()      CPUID+RDTSC (run boundary)
|                          rdtsc_fast()            bare RDTSC (hot-path stamps)
|                          calibrate_cpu_frequency / cycles_to_ns
+-- cpu_utils.h/.cpp  -- CPU affinity and hardware spin
|                          pinThreadToCore / isCoreIsolated / cpu_pause
+-- ring_buffer.h     -- Lock-free SPSC ring buffer template
|                          RingBuffer<T, Size>  push / pop / pre_fault_memory
+-- messages.h/.cpp   -- Shared message types and global queues
|                          UpdateMessage, PacketView
|                          updateBuffer (engine -> publisher)
|                          inputQueue   (network  -> engine)
|                          Lifecycle atomics: stopPublisher, stopNetworkThread,
|                            stopEngine, startNetworkTraffic
+-- order_book.h/.cpp -- Price-Time FIFO matching engine
|                          addOrder / cancelOrder / executeTrade
|                          modifyOrder / executeMarketOrder
|                          saveSnapshot / loadSnapshot
|                          getVWAP / getTotalTradedVolume
+-- journal.h         -- Memory-mapped sequential event log
|                          MappedJournal<T, MaxEntries>
+-- latency.h/.cpp    -- Latency measurement and reporting
|                          LatencyHistogram (10 ns buckets)
|                          LatencyRecorder  (P50/P90/P99/P99.9 + spike report)
+-- threads.h/.cpp    -- Thread entry points
|                          networkThread   -- zero-copy packet producer
|                          publisherThread -- update consumer + journal writer
+-- main.cpp          -- Engine entry point
|                          startup -> pre-fault -> warmup -> benchmark -> shutdown
+-- data_gen.cpp      -- Synthetic market-data generator (market_data.bin)
+-- CMakeLists.txt    -- Release build: -O3 -march=native -mtune=native + LTO
```

---

## Build & Run

### Prerequisites

- GCC 9+ or Clang 10+ with C++17 support
- CMake 3.16+
- Linux x86-64 (required for RDTSC/RDTSCP and pthread_setaffinity_np)

### Steps

```bash
# 1. Generate synthetic test data (~100 K messages)
g++ -O2 -std=c++17 data_gen.cpp -o data_gen && ./data_gen

# 2. Configure and build (Release mode)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 3. Run from the repo root so the binary can find market_data.bin
cd ..
./build/orderbook
```

Output files created in the working directory:

| File | Description |
|------|-------------|
| `event_journal.bin`      | Sequential log of all published events (~300 MB for 10 M slots) |
| `orderbook_snapshot.bin` | Full book state at shutdown (~34 MB) |

### Optional: CPU Isolation for Deterministic Latency

Add to `/etc/default/grub` to dedicate core 2 entirely to the engine:

```
GRUB_CMDLINE_LINUX="isolcpus=2 nohz_full=2 rcu_nocbs=2"
```

Then:

```bash
sudo update-grub && sudo reboot
```

| Flag | Effect |
|------|--------|
| `isolcpus=2`  | Removes core from scheduler load balancing |
| `nohz_full=2` | Disables periodic timer ticks (tickless mode) |
| `rcu_nocbs=2` | Offloads RCU callbacks to other cores |

Expected effect: max spike drops from ~5 us -> < 1 us; P99.9 becomes
consistently sub-microsecond.

---

## Key Implementation Techniques

### 1. Zero-Copy Networking

The network thread publishes a `PacketView` -- a 16-byte struct holding a raw
pointer into the memory-mapped file and a cached type byte. The engine casts that
pointer directly to the wire-format struct, with no intermediate copy.

```
Before (naive):    network -> memcpy 136 bytes -> queue -> engine
After (zero-copy): network -> 16-byte pointer  -> queue -> engine casts in-place
```

Four `PacketView` structs fit in a single 64-byte cache line.

### 2. Lock-Free SPSC Ring Buffer

Both queues are single-producer / single-consumer ring buffers with `std::atomic`
indices. Key design choices:

- `alignas(64)` on each index group -- prevents false sharing between producer and
  consumer variables across cores
- Cached remote indices -- each thread keeps a local copy of the other thread's
  index; the `acquire` load only fires on the slow path
- Power-of-2 capacity -- index wrap uses bitwise `& MASK` instead of `% Size`

### 3. Intrusive FIFO Linked List per Price Level

Each `PriceLevel` holds `{ head, tail, total_qty }`. Order IDs serve as indices
into the flat `order_lookup` array; `OrderInfo::next_id` chains nodes within a
level. Enqueue is a tail-pointer update; dequeue is a head-pointer advance.
No heap allocation is required -- all nodes live in a pre-allocated `std::array`.

### 4. Memory Pre-Faulting

Every ring buffer and order table is touched once per 4 KB page at startup.
This maps all physical pages upfront, removing demand-paging latency spikes
(which can exceed 100 us on first access to a cold page) from the measured path.

### 5. Warmup Synchronisation Barrier

The network thread blocks on `startNetworkTraffic` until the engine finishes
pre-faulting and warmup. This ensures the benchmark measures real inter-core
contention rather than draining a pre-populated queue.

### 6. Dead-Store Elimination Guard

Without an observable side effect, the compiler can prove (via LTO) that
`UpdateMessage` fields are never read and strip the writes from `addOrder`. A
`volatile` accumulator in the publisher thread forces those fields to be treated
as externally observable.

### 7. Bitmap O(1) Best Bid / Ask

Price occupancy is tracked in `bid_bitmap` and `ask_bitmap` -- one bit per price
level. Best-price search uses `__builtin_ctzll` (count trailing zeros), a single
hardware instruction:

```cpp
// O(1) level deactivation
bid_bitmap[price / 64] &= ~(1ULL << (price % 64));

// O(1) best-bid search
int best = idx * 64 + __builtin_ctzll(bid_bitmap[idx]);
```

A linear scan over a sparse book costs O(n) cycles; the bitmap approach is
constant time regardless of the number of active price levels.

### 8. RDTSC Timing Sandwich

Three measurement functions serve different roles:

| Function | Instructions | Overhead | Use case |
|----------|-------------|----------|----------|
| `rdtsc_start_full()` | CPUID + RDTSC   | ~150 cycles | Outer run boundary (once) |
| `rdtsc_start()`      | LFENCE + RDTSC  | ~20 cycles  | Per-message start |
| `rdtsc_end()`        | RDTSCP + LFENCE | ~20 cycles  | Per-message end |
| `rdtsc_fast()`       | RDTSC           | ~2 cycles   | Hot-path event timestamps |

`LFENCE` drains pending loads before sampling. `RDTSCP` waits for prior
instructions to retire. `CPUID` fully serialises the pipeline but costs
~100-150 cycles -- appropriate for a one-time boundary, not per-message.
