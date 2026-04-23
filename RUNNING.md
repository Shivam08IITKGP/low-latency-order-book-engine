# Low-Latency Order Book Engine — Running Guide

## Architecture

```
Core 0 (Network Thread)   →  inputQueue (SPSC Ring)  →  Core 2 (Engine Thread)
Core 2 (Engine Thread)    →  updateBuffer (SPSC Ring) →  Core 3 (Publisher Thread)
```

| Thread    | Core | Isolated? | Role                              |
|-----------|------|-----------|-----------------------------------|
| Engine    | 2    | ✅ YES    | Matching, hot path                |
| Network   | 0    | ❌ NO     | Packet producer (file replay)     |
| Publisher | 3    | ❌ NO     | Persistence & downstream fan-out  |

> **To reach 15M+ msgs/sec:** Isolate cores 0 and 3 as well (see GRUB section below).

---

## Prerequisites

```bash
# Build tools
sudo apt install g++ make

# Python (optional, for multi-run stats)
pip install statistics  # stdlib, no install needed
```

---

## Step 1: Verify CPU Isolation

Core 2 must be isolated. Check:

```bash
cat /sys/devices/system/cpu/isolated
# Expected output: 2

cat /proc/cmdline | grep isolcpus
# Expected: isolcpus=2 nohz_full=2 rcu_nocbs=2
```

If Core 2 is NOT isolated, add it to GRUB (see below).

---

## Step 2: Build

```bash
make clean && make -j$(nproc)
```

This produces two binaries:
- `orderbook` — the engine
- `data_gen`  — the market data generator

---

## Step 3: Generate Test Data

```bash
# Quick test (fits in L3 cache, ~13M msgs/sec)
./data_gen 500000

# Full benchmark (10M orders, cache-miss regime)
./data_gen 10000000
```

---

## Step 4: Run the Engine

```bash
./orderbook
```

The engine will print:
1. CPU frequency calibration
2. Core isolation status for each thread
3. Warmup progress
4. Throughput + full latency histogram + percentile breakdown

---

## Step 5: Automated Multi-Run Stats (Optional)

The `run_stats.py` script runs the engine 20 times and reports median throughput and latency percentiles:

```bash
# NOTE: run_stats.py is hardcoded to ./build/orderbook — fix the path first:
sed -i "s|./build/orderbook|./orderbook|g" run_stats.py

python3 run_stats.py
```

---

## GRUB: CPU Isolation Setup

Edit GRUB config:

```bash
sudo nano /etc/default/grub
```

Change the `GRUB_CMDLINE_LINUX` line. For Core 2 only (current):
```
GRUB_CMDLINE_LINUX="isolcpus=2 nohz_full=2 rcu_nocbs=2"
```

For full pipeline isolation (recommended for max throughput):
```
GRUB_CMDLINE_LINUX="isolcpus=0,2,3 nohz_full=0,2,3 rcu_nocbs=0,2,3"
```

Apply:
```bash
sudo update-grub && sudo reboot
```

After reboot, verify:
```bash
cat /sys/devices/system/cpu/isolated
# Should show: 0,2,3
```

---

## Performance Reference (2.5 GHz laptop, Core 2 isolated only)

| Data Set | Throughput     | Mean Latency | P99 Latency |
|----------|----------------|--------------|-------------|
| 500k msg | ~13.6M msgs/s  | ~48 ns       | ~115 ns     |
| 10M msg  | ~6M msgs/s     | ~60 ns       | ~115 ns     |

The 10M drop is due to the order pool (160 MB) exceeding L3 cache.
With 0,2,3 isolated: expect 15–18M msgs/sec at 500k.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `[WARNING: NOT ISOLATED]` on Core 2 | GRUB not updated | See GRUB section |
| Segfault on 10M run | Old binary with 1M order_lookup | `make clean && make` |
| Low throughput (<5M) | Core 0 not isolated | Isolate 0,2,3 in GRUB |
| Interleaved console output | Multiple threads printing simultaneously | Cosmetic only — results are correct |
