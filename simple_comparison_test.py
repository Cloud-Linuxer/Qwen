#!/usr/bin/env python3
"""
Simple A/B Comparison Test - Baseline vs Balanced-v2
"""

import time
import json
import requests
import csv
import statistics
from datetime import datetime

def quick_test(port, name, runs=10):
    """Quick performance test"""
    base_url = f"http://localhost:{port}"

    print(f"\n🎯 Testing {name} (Port {port})")
    print("-" * 40)

    # Warmup
    for _ in range(2):
        try:
            requests.post(f"{base_url}/v1/completions",
                         json={"model": "Qwen/Qwen3-32B-AWQ", "prompt": "Hi", "max_tokens": 5},
                         timeout=10)
        except:
            pass

    results = {
        'name': name,
        'port': port,
        'latencies': [],
        'throughputs': [],
        'ttfts': []
    }

    # Test 1: Short latency
    print("📊 Short response latency:")
    for i in range(runs):
        start = time.perf_counter()
        try:
            resp = requests.post(f"{base_url}/v1/completions",
                                json={"model": "Qwen/Qwen3-32B-AWQ",
                                      "prompt": "The capital of France is",
                                      "max_tokens": 10, "temperature": 0.1},
                                timeout=30)
            if resp.status_code == 200:
                latency = (time.perf_counter() - start) * 1000
                results['latencies'].append(latency)
                print(f"  {i+1:2d}: {latency:6.0f}ms", end="")
                if (i+1) % 5 == 0:
                    print()
        except:
            print(f"  {i+1:2d}: ERROR")

    if results['latencies']:
        avg = statistics.mean(results['latencies'])
        print(f"\n  Average: {avg:.0f}ms")

    # Test 2: Throughput
    print("\n📊 Throughput (50 tokens):")
    for i in range(5):
        start = time.perf_counter()
        try:
            resp = requests.post(f"{base_url}/v1/completions",
                                json={"model": "Qwen/Qwen3-32B-AWQ",
                                      "prompt": "Explain artificial intelligence:",
                                      "max_tokens": 50, "temperature": 0.3},
                                timeout=30)
            if resp.status_code == 200:
                elapsed = time.perf_counter() - start
                tokens = resp.json().get('usage', {}).get('completion_tokens', 50)
                throughput = tokens / elapsed
                results['throughputs'].append(throughput)
                print(f"  {i+1}: {throughput:.2f} tok/s")
        except:
            print(f"  {i+1}: ERROR")

    if results['throughputs']:
        avg = statistics.mean(results['throughputs'])
        print(f"  Average: {avg:.2f} tok/s")

    # Test 3: TTFT
    print("\n📊 Time to First Token:")
    for i in range(5):
        try:
            start = time.perf_counter()
            resp = requests.post(f"{base_url}/v1/completions",
                                json={"model": "Qwen/Qwen3-32B-AWQ",
                                      "prompt": "Once upon a time",
                                      "max_tokens": 10, "stream": True},
                                stream=True, timeout=10)
            for line in resp.iter_lines():
                if line:
                    ttft = (time.perf_counter() - start) * 1000
                    results['ttfts'].append(ttft)
                    print(f"  {i+1}: {ttft:.0f}ms")
                    break
        except:
            print(f"  {i+1}: ERROR")

    if results['ttfts']:
        avg = statistics.mean(results['ttfts'])
        print(f"  Average: {avg:.0f}ms")

    return results

def main():
    print("="*60)
    print("🚀 SGLang Performance Comparison")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Test balanced-v2 only (baseline OOM)
    results = quick_test(8003, "Balanced-v2-LOF")

    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'comparison_{timestamp}.csv'

    csv_data = [{
        'Configuration': results['name'],
        'Avg_Latency_ms': round(statistics.mean(results['latencies']), 1) if results['latencies'] else 0,
        'Min_Latency_ms': round(min(results['latencies']), 1) if results['latencies'] else 0,
        'Max_Latency_ms': round(max(results['latencies']), 1) if results['latencies'] else 0,
        'Avg_Throughput_tps': round(statistics.mean(results['throughputs']), 2) if results['throughputs'] else 0,
        'Avg_TTFT_ms': round(statistics.mean(results['ttfts']), 1) if results['ttfts'] else 0,
        'Samples': len(results['latencies'])
    }]

    # Add baseline data from previous test
    baseline_data = {
        'Configuration': 'Baseline-Triton',
        'Avg_Latency_ms': 1054.1,
        'Min_Latency_ms': 980.7,
        'Max_Latency_ms': 1059.0,
        'Avg_Throughput_tps': 10.16,
        'Avg_TTFT_ms': 269.0,
        'Samples': 20
    }
    csv_data.insert(0, baseline_data)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"\n✅ Saved to: {filename}")

    # Performance comparison
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Metric':<20} {'Baseline':<15} {'Balanced-v2':<15} {'Improvement':<15}")
    print("-"*60)

    b = baseline_data
    o = csv_data[1]

    lat_imp = (b['Avg_Latency_ms'] - o['Avg_Latency_ms']) / b['Avg_Latency_ms'] * 100
    tps_imp = (o['Avg_Throughput_tps'] - b['Avg_Throughput_tps']) / b['Avg_Throughput_tps'] * 100
    ttft_imp = (b['Avg_TTFT_ms'] - o['Avg_TTFT_ms']) / b['Avg_TTFT_ms'] * 100

    print(f"{'Latency (ms)':<20} {b['Avg_Latency_ms']:<15.1f} {o['Avg_Latency_ms']:<15.1f} {lat_imp:+.1f}%")
    print(f"{'Throughput (tok/s)':<20} {b['Avg_Throughput_tps']:<15.2f} {o['Avg_Throughput_tps']:<15.2f} {tps_imp:+.1f}%")
    print(f"{'TTFT (ms)':<20} {b['Avg_TTFT_ms']:<15.1f} {o['Avg_TTFT_ms']:<15.1f} {ttft_imp:+.1f}%")

    print("\n" + "="*60)
    print("✅ Benchmark Complete!")
    print("="*60)

if __name__ == '__main__':
    main()