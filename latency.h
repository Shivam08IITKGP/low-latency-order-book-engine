#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <array>
#include <string>

// --------------------------------------------------------------------
// LATENCY HISTOGRAM
// --------------------------------------------------------------------
// Buckets latency samples into fixed-width bins and prints a visual bar
// chart plus min/max/jitter summary.  Bucket width defaults to 10 ns.
class LatencyHistogram
{
private:
    std::vector<uint64_t> buckets;
    const uint64_t        bucket_size_ns;
    uint64_t              max_latency   = 0;
    uint64_t              min_latency   = UINT64_MAX;
    uint64_t              total_samples = 0;

public:
    explicit LatencyHistogram(uint64_t bucket_size = 10)
        : bucket_size_ns(bucket_size)
    {
        // Buckets: 0–10 ns, 10–20 ns, … , 990–1000 ns, 1000+ ns
        buckets.resize(101, 0);
    }

    void addSample(uint64_t latency_ns)
    {
        total_samples++;
        max_latency = std::max(max_latency, latency_ns);
        min_latency = std::min(min_latency, latency_ns);

        size_t bucket = std::min(latency_ns / bucket_size_ns,
                                 static_cast<uint64_t>(buckets.size() - 1));
        buckets[bucket]++;
    }

    void printHistogram() const
    {
        std::cout << "\n========== LATENCY HISTOGRAM ==========\n";
        std::cout << "Min: " << min_latency << " ns\n";
        std::cout << "Max: " << max_latency
                  << " ns (JITTER: " << (max_latency - min_latency) << " ns)\n";
        std::cout << "Bucket Size: " << bucket_size_ns << " ns\n\n";

        for (size_t i = 0; i < buckets.size(); i++)
        {
            if (buckets[i] == 0)
                continue;

            uint64_t range_start = i * bucket_size_ns;
            uint64_t range_end   = (i == buckets.size() - 1)
                                       ? UINT64_MAX
                                       : (i + 1) * bucket_size_ns;

            double percentage = (100.0 * buckets[i]) / total_samples;

            std::cout << std::setw(6) << range_start << "-"
                      << std::setw(6)
                      << (range_end == UINT64_MAX ? "∞" : std::to_string(range_end))
                      << " ns: " << std::setw(8) << buckets[i]
                      << " (" << std::fixed << std::setprecision(2)
                      << percentage << "%)";

            int bar = static_cast<int>(percentage / 2);
            std::cout << " ";
            for (int j = 0; j < bar; j++)
                std::cout << "█";
            std::cout << "\n";
        }

        std::cout << "\nTotal Samples: " << total_samples << "\n";
        std::cout << "=======================================\n";
    }

    void printJitterAnalysis() const
    {
        uint64_t jitter = max_latency - min_latency;
        std::cout << "\n========== JITTER ANALYSIS ==========\n";
        std::cout << "Jitter (Max - Min): " << jitter << " ns\n";

        if      (jitter < 1000)  std::cout << "✓ Excellent: Very low jitter\n";
        else if (jitter < 5000)  std::cout << "✓ Good: Acceptable jitter\n";
        else if (jitter < 10000) std::cout << "⚠ Warning: High jitter detected\n";
        else                     std::cout << "✗ Critical: Very high jitter (possible cache miss/page fault)\n";

        std::cout << "=====================================\n";
    }
};

// --------------------------------------------------------------------
// LATENCY RECORDER
// --------------------------------------------------------------------
// Zero-overhead per-message recorder: writes into a flat pre-allocated
// array from the hot path (no heap allocation, no branching overhead).
// After the run, printReport() sorts and computes full percentile stats.
class LatencyRecorder
{
private:
    struct alignas(16) Sample
    {
        uint64_t latency_ns;
        uint32_t index;
        char     type;
    };

    static constexpr size_t MAX_SAMPLES = 10000005;
    std::array<Sample, MAX_SAMPLES> samples{};
    size_t count = 0;

public:
    // Inline, zero-branch hot-path write (with safety check)
    inline void addSample(uint64_t latency_ns, char type, uint32_t index)
    {
        if (count < MAX_SAMPLES)
            samples[count++] = {latency_ns, index, type};
    }

    void printReport() const
    {
        if (count == 0) return;

        uint64_t spikes_over_1us   = 0;
        uint64_t spikes_over_10us  = 0;
        uint64_t spikes_over_100us = 0;

        std::vector<uint64_t> vals;
        vals.reserve(count);
        uint64_t sum = 0;

        for (size_t i = 0; i < count; ++i)
        {
            uint64_t v = samples[i].latency_ns;
            vals.push_back(v);
            sum += v;
            if (v >   1000) spikes_over_1us++;
            if (v >  10000) spikes_over_10us++;
            if (v > 100000) spikes_over_100us++;
        }

        double mean = static_cast<double>(sum) / count;
        double var  = 0.0;
        for (auto v : vals) { double d = static_cast<double>(v) - mean; var += d * d; }
        var  /= count;
        double stddev = std::sqrt(var);

        std::sort(vals.begin(), vals.end());
        auto pct = [&](double p) -> uint64_t {
            size_t idx = static_cast<size_t>(std::ceil((p / 100.0) * count)) - 1;
            if (idx >= vals.size()) idx = vals.size() - 1;
            return vals[idx];
        };

        std::cout << "\n========== DETAILED LATENCY REPORT ==========\n";
        std::cout << "Samples: " << count << "\n";
        std::cout << "Mean: "   << static_cast<uint64_t>(mean)   << " ns"
                  << "  StdDev: " << static_cast<uint64_t>(stddev) << " ns\n";
        std::cout << "P50: "  << pct(50.0)  << " ns"
                  << "  P90: "  << pct(90.0)  << " ns"
                  << "  P99: "  << pct(99.0)  << " ns"
                  << "  P99.9: "<< pct(99.9)  << " ns\n";
        std::cout << "Spikes >1us: "   << spikes_over_1us
                  << "  >10us: "        << spikes_over_10us
                  << "  >100us: "       << spikes_over_100us << "\n";

        // Top 10 latency spikes with message context
        std::vector<size_t> idxs(count);
        for (size_t i = 0; i < count; ++i) idxs[i] = i;
        std::partial_sort(idxs.begin(),
                          idxs.begin() + std::min<size_t>(10, idxs.size()),
                          idxs.end(),
                          [&](size_t a, size_t b) {
                              return samples[a].latency_ns > samples[b].latency_ns;
                          });

        std::cout << "Top spikes (latency ns, sample index, msg type):\n";
        for (size_t i = 0; i < std::min<size_t>(10, idxs.size()); ++i)
        {
            const auto& s = samples[idxs[i]];
            std::cout << "  " << s.latency_ns
                      << " ns, idx=" << s.index
                      << ", type=" << s.type << "\n";
        }

        std::cout << "============================================\n";
    }
};

// Global latency recorder instance (written from engine hot path)
extern LatencyRecorder g_latency_recorder;