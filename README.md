# Low-Latency Order Book Engine

A production-grade, multi-threaded C++ order book engine built around zero-copy networking, lock-free inter-thread communication, hardware-level timing, and CPU affinity. Designed to reflect real-world techniques used in latency-sensitive trading systems.

---

## Performance (5-Run Benchmark, Non-Isolated Core)

All results measured with 100,001 messages on a standard Linux desktop — no kernel CPU isolation.

| Run | Throughput | Min    | P50   | P90    | P99    | P99.9  |
|-----|-----------|--------|-------|--------|--------|--------|
| 1   | 6.25M/s   | 45 ns  | 89 ns | 104 ns | 116 ns | 249 ns |
| 2   | 7.80M/s   | 47 ns  | 56 ns | 91 ns  | 118 ns | 198 ns |
| 3   | 6.14M/s   | 48 ns  | 95 ns | 111 ns | 122 ns | 275 ns |
| 4   | 5.95M/s   | 51 ns  | 92 ns | 106 ns | 128 ns | 276 ns |
| 5   | 6.97M/s   | 51 ns  | 91 ns | 112 ns | 126 ns | 239 ns |
| **Avg** | **6.62M/s** | **~48 ns** | **~85 ns** | **~105 ns** | **~122 ns** | **~247 ns** |

**Timing method:** LFENCE+RDTSC (start) / RDTSCP+LFENCE (end) — per-message intel recommended hardware TSC sandwich  
**CPU frequency:** ~2497 MHz (calibrated at startup via TSC)

> P99.9 variance is caused by the OS scheduler interrupting the non-isolated engine core.  
> With `isolcpus=2 nohz_full=2 rcu_nocbs=2`, P99.9 is expected to drop below 5 µs consistently.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Order Book Engine — Thread Pipeline        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Network Thread  (Core 0)                               │
│  ─────────────────────────────────────────────────────  │
│  Reads memory-mapped file, builds zero-copy PacketViews │
│  (16-byte pointer+type, not a 136-byte payload copy)    │
│      │                                                  │
│      ▼  Lock-Free SPSC Queue  (524 K slots)             │
│                                                         │
│  Engine Thread  (Core 2)                                │
│  ─────────────────────────────────────────────────────  │
│  Pops PacketViews, casts payload pointer directly       │
│  Dispatches: addOrder / cancelOrder / executeTrade /    │
│              modifyOrder / executeMarketOrder           │
│  Per-message RDTSC sandwich timing                      │
│      │                                                  │
│      ▼  Lock-Free Ring Buffer  (1 M slots)              │
│                                                         │
│  Publisher Thread  (Core 3)                             │
│  ─────────────────────────────────────────────────────  │
│  Drains UpdateMessages; in production this thread       │
│  would fan out to market-data feeds, client             │
│  connections, and a persistence layer                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
.
├── common.h          # Wire-format structs: StreamHeader, OrderMessage, TradeMessage
│
├── timing.h/.cpp     # Hardware TSC timing utilities
│                     #   rdtsc_start_full() — CPUID+RDTSC  (outer boundary, once per run)
│                     #   rdtsc_start()      — LFENCE+RDTSC (per-message start)
│                     #   rdtsc_end()        — RDTSCP+LFENCE (per-message end)
│                     #   rdtsc_fast()       — bare RDTSC for hot-path timestamps
│                     #   calibrate_cpu_frequency() / cycles_to_ns() / get_timestamp_ns()
│
├── cpu_utils.h/.cpp  # CPU affinity and hardware spin
│                     #   pinThreadToCore() / isCoreIsolated() / cpu_pause()
│
├── ring_buffer.h     # Lock-free SPSC ring buffer template
│                     #   RingBuffer<T, Size>  push / pop / pre_fault_memory()
│
├── messages.h/.cpp   # Shared message types and global queues
│                     #   UpdateMessage, PacketView
│                     #   updateBuffer (engine → publisher), inputQueue (network → engine)
│                     #   stopPublisher / stopNetworkThread / stopEngine
│                     #   startNetworkTraffic — warmup synchronisation barrier
│
├── order_book.h/.cpp # Core price-level matching engine
│                     #   addOrder / cancelOrder / executeTrade
│                     #   modifyOrder / executeMarketOrder
│
├── latency.h/.cpp    # Latency measurement and reporting
│                     #   LatencyHistogram / LatencyRecorder / g_latency_recorder
│
├── threads.h/.cpp    # Thread entry points
│                     #   networkThread()   — zero-copy packet producer
│                     #   publisherThread() — update consumer
│
├── main.cpp          # Engine entry point
│                     #   startup → pre-fault → warmup → benchmark loop → shutdown
│
├── data_gen.cpp      # Synthetic market-data generator (produces market_data.bin)
│
└── CMakeLists.txt    # Release build configuration (-O3 -march=native -mtune=native)
```

---

## Build & Run

### Prerequisites

- GCC 9+ or Clang 10+ with C++17 support
- CMake 3.16+
- Linux x86-64 (required for `RDTSC`/`RDTSCP` and `pthread_setaffinity_np`)

### Steps

```bash
# 1. Generate synthetic test data (market_data.bin, ~100 k messages)
g++ -O2 -std=c++17 data_gen.cpp -o data_gen && ./data_gen

# 2. Configure and build (Release mode)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 3. Run from repo root so the binary can find market_data.bin
cd ..
./build/orderbook
```

### Optional: CPU Isolation for Deterministic Latency

Add to `/etc/default/grub` to dedicate core 2 entirely to the engine:

```
GRUB_CMDLINE_LINUX="isolcpus=2 nohz_full=2 rcu_nocbs=2"
```

Then:
```bash
sudo update-grub && sudo reboot
```

- `isolcpus=2`  — removes core from scheduler load balancing
- `nohz_full=2` — disables periodic timer ticks (tickless mode)
- `rcu_nocbs=2` — offloads RCU callbacks to other cores

Expected effect: max latency drops from ~200 µs → < 5 µs; P99.9 becomes consistently sub-microsecond.

---

## Key Techniques

### 1. Zero-Copy Networking

Instead of copying each packet payload (136 bytes) into the queue, the network thread publishes a `PacketView` — a 16-byte struct holding a raw pointer into the memory-mapped file and a cached type byte. The engine casts that pointer directly to the wire-format struct.

```
Before:  network → memcpy 136 bytes → queue → engine
After:   network → 16-byte pointer view → queue → engine casts in-place
```

- 8.5× fewer bytes transferred between cores per message
- 4 `PacketView`s fit in a single 64-byte cache line

### 2. Lock-Free SPSC Ring Buffer

Both queues are single-producer / single-consumer ring buffers backed by `std::atomic` indices.

Critical design choices:
- **`alignas(64)` on each index group** — prevents producer and consumer index variables sharing a cache line (false sharing would cause cross-core coherency traffic on every access)
- **Cached remote indices** — each thread keeps a local copy of the other thread's index; the `acquire` load only fires on the slow path when the cached value indicates full/empty
- **Power-of-2 capacity** — index wrap uses `& MASK` instead of `% Size`

### 3. Memory Pre-Faulting

Every ring buffer and order table is touched once per 4 KB page at startup before any benchmark timing begins. This maps all physical pages up front, removing demand-paging latency spikes (which can exceed 100 µs on the first access to a cold page) from the measured path.

### 4. Warmup Synchronisation Barrier (`startNetworkTraffic`)

The network thread blocks on an atomic flag until the engine finishes its warmup. Without this, the network thread saturates the input queue during pre-faulting, and the first benchmark measurements see a fully-hot, pre-populated queue — not real inter-core contention.

```cpp
// main.cpp — taken immediately before the engine loop
uint64_t totalStart = rdtsc_start();
startNetworkTraffic.store(true, std::memory_order_release);
```

### 5. Dead-Store Elimination Guard

Without an observable side effect in the publisher thread, the compiler can prove (across translation units with LTO) that `UpdateMessage` fields written by the engine are never read, and strip the writes from `addOrder`. A `volatile` accumulator forces those fields to be treated as externally observable:

```cpp
volatile uint64_t dummy_accumulator = 0;
// inside the pop loop:
dummy_accumulator += msg.price;
```

### 6. RDTSC/RDTSCP Timing Sandwich

Three functions serve different measurement roles:

| Function | Instructions | Overhead | Use case |
|---|---|---|---|
| `rdtsc_start_full()` | CPUID + RDTSC | ~150 cycles | Outer run boundary (called once) |
| `rdtsc_start()` | LFENCE + RDTSC | ~20 cycles | Per-message start |
| `rdtsc_end()` | RDTSCP + LFENCE | ~20 cycles | Per-message end |
| `rdtsc_fast()` | RDTSC | ~2 cycles | Hot-path event timestamps |

`LFENCE` drains pending load operations before reading the counter. `RDTSCP` waits for prior instructions to retire before sampling. `CPUID` fully serialises the pipeline but costs ~100–150 cycles — appropriate for a one-time benchmark boundary, not per-message measurement.

### 7. Bitmap O(1) Best Bid / Ask

Price occupancy is tracked in `bid_bitmap` and `ask_bitmap` — one bit per price level. Finding the best price uses a hardware count-trailing-zeros instruction:

```cpp
// O(1) clear when a level depletes
bid_bitmap[price / 64] &= ~(1ULL << (price % 64));

// O(1) best-price discovery — single CPU instruction
int best = idx * 64 + __builtin_ctzll(bid_bitmap[idx]);
```

A linear scan on a sparse book can cost thousands of cycles; the bitmap approach is constant time regardless of book depth.

### 8. Compact OrderInfo Bitfield

Per-order metadata fits in 8 bytes via a bitfield:

```cpp
struct alignas(8) OrderInfo {
    uint64_t price    : 32;  // covers the full price range
    uint64_t quantity : 31;  // up to ~2 billion shares
    uint64_t side     :  1;  // 0 = Buy, 1 = Sell
};
```

This halves the memory footprint of the 1 M-entry order table (8 MB vs 16 MB), doubling cache density and halving TLB pressure.

### 9. CPU Pinning

Each thread is bound to a fixed core via `pthread_setaffinity_np`:

- **Core 0** — network thread
- **Core 2** — engine thread (hot path)
- **Core 3** — publisher thread

This prevents the OS scheduler from migrating threads mid-run, which would invalidate L1/L2 caches and introduce unpredictable latency spikes. The engine also checks whether the assigned core is kernel-isolated at startup and prints a warning if not.

---

## Latency Instrumentation

TSC start/end values for every message are stored into exact-capacity `std::vector` buffers (sized by a pre-scan of the message file). After the engine loop exits, TSC deltas are converted to nanoseconds and analysed by:

- **`LatencyHistogram`** — 10 ns bucket histogram, min/max/jitter summary
- **`LatencyRecorder`** — per-sample store with P50 / P90 / P99 / P99.9, mean, stddev, and top-10 spike report (sample index + message type)

All post-processing runs after the benchmark timer stops, contributing zero overhead to the measured path.


