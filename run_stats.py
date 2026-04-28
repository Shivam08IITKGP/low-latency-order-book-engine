import subprocess
import re
import statistics

throughputs = []
p50s = []
p90s = []
p99s = []
p999s = []
mins = []
maxs = []

def extract(line, pattern):
    match = re.search(pattern, line)
    return float(match.group(1)) if match else None

for i in range(20):
    result = subprocess.run(['./orderbook_lto'], capture_output=True, text=True)
    output = result.stdout
    
    tp = extract(output, r'Throughput:\s+([\d.]+)\s+msgs/sec')
    if tp: throughputs.append(tp / 1e6) # M ops/s
    
    m = re.search(r'P50:\s+(\d+)\s+ns\s+P90:\s+(\d+)\s+ns\s+P99:\s+(\d+)\s+ns\s+P99\.9:\s+(\d+)\s+ns', output)
    if m:
        p50s.append(float(m.group(1)))
        p90s.append(float(m.group(2)))
        p99s.append(float(m.group(3)))
        p999s.append(float(m.group(4)))

    m_min = re.search(r'Min:\s+(\d+)\s+ns', output)
    if m_min: mins.append(float(m_min.group(1)))
    
    m_max = re.search(r'Max:\s+(\d+)\s+ns', output)
    if m_max: maxs.append(float(m_max.group(1)))

print("\n--- Results (Throughput in M ops/s, Latency in ns) ---")
print(f"{'Metric':<15} | {'Value':<10}")
print("-" * 28)
if throughputs:
    print(f"{'Mean TP':<15} | {statistics.mean(throughputs):.4f}")
    print(f"{'Median TP':<15} | {statistics.median(throughputs):.4f}")
    if len(throughputs) > 1:
        print(f"{'StdDev TP':<15} | {statistics.stdev(throughputs):.4f}")
    print(f"{'Min TP':<15} | {min(throughputs):.4f}")
    print(f"{'Max TP':<15} | {max(throughputs):.4f}")

if p50s: print(f"{'Median P50':<15} | {statistics.median(p50s):.2f}")
if p90s: print(f"{'Median P90':<15} | {statistics.median(p90s):.2f}")
if p99s: print(f"{'Median P99':<15} | {statistics.median(p99s):.2f}")
if p999s: print(f"{'Median P99.9':<15} | {statistics.median(p999s):.2f}")
if mins: print(f"{'Median Min':<15} | {statistics.median(mins):.2f}")
if maxs: print(f"{'Median Max':<15} | {statistics.median(maxs):.2f}")

if len(throughputs) > 1:
    cv = (statistics.stdev(throughputs) / statistics.mean(throughputs)) * 100
    print(f"\nCoefficient of Variation (CV): {cv:.2f}%")
    if cv > 1.0:
        print("Conclusion: 0.5-2% differences are likely NOT distinguishable from noise (CV > 1%).")
    else:
        print("Conclusion: 0.5-2% differences may be distinguishable from noise (CV < 1%).")
