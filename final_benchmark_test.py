#!/usr/bin/env python3
"""
Final Performance Benchmark - Baseline vs Balanced-v2
"""

import time
import json
import requests
import csv
import statistics
from datetime import datetime
import subprocess

def test_configuration(port, name, num_tests=20):
    """Test a single configuration"""
    base_url = f"http://localhost:{port}"
    results = {
        'name': name,
        'port': port,
        'short_latencies': [],
        'medium_latencies': [],
        'medium_throughputs': [],
        'ttfts': [],
        'korean_throughputs': []
    }

    # Warmup
    print(f"🔥 Warming up {name}...")
    for _ in range(3):
        try:
            requests.post(
                f"{base_url}/v1/completions",
                json={"model": "Qwen/Qwen3-32B-AWQ", "prompt": "Hi", "max_tokens": 5},
                timeout=10
            )
            time.sleep(0.5)
        except:
            pass

    # Test 1: Short response latency (20 runs)
    print(f"📊 Testing short response latency...")
    for i in range(num_tests):
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "model": "Qwen/Qwen3-32B-AWQ",
                    "prompt": "The capital of France is",
                    "max_tokens": 10,
                    "temperature": 0.1
                },
                timeout=30
            )
            if response.status_code == 200:
                latency = (time.perf_counter() - start) * 1000
                results['short_latencies'].append(latency)
                print(f"  Run {i+1}/{num_tests}: {latency:.0f}ms")
        except Exception as e:
            print(f"  Error: {e}")

    # Test 2: Medium response throughput (10 runs)
    print(f"📊 Testing medium response throughput...")
    for i in range(10):
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "model": "Qwen/Qwen3-32B-AWQ",
                    "prompt": "Write about artificial intelligence and its impact:",
                    "max_tokens": 50,
                    "temperature": 0.3
                },
                timeout=30
            )
            if response.status_code == 200:
                elapsed = (time.perf_counter() - start)
                data = response.json()
                tokens = data.get('usage', {}).get('completion_tokens', 50)
                throughput = tokens / elapsed
                results['medium_latencies'].append(elapsed * 1000)
                results['medium_throughputs'].append(throughput)
                print(f"  Run {i+1}/10: {throughput:.2f} tok/s")
        except Exception as e:
            print(f"  Error: {e}")

    # Test 3: TTFT (10 runs)
    print(f"📊 Testing Time to First Token...")
    for i in range(10):
        try:
            start = time.perf_counter()
            response = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "model": "Qwen/Qwen3-32B-AWQ",
                    "prompt": "Once upon a time:",
                    "max_tokens": 20,
                    "temperature": 0.5,
                    "stream": True
                },
                stream=True,
                timeout=10
            )
            for line in response.iter_lines():
                if line:
                    ttft = (time.perf_counter() - start) * 1000
                    results['ttfts'].append(ttft)
                    print(f"  Run {i+1}/10: {ttft:.0f}ms")
                    break
        except Exception as e:
            print(f"  Error: {e}")

    # Test 4: Korean processing (5 runs)
    print(f"📊 Testing Korean language processing...")
    for i in range(5):
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "model": "Qwen/Qwen3-32B-AWQ",
                    "prompt": "인공지능의 미래에 대해 설명해주세요:",
                    "max_tokens": 30,
                    "temperature": 0.3
                },
                timeout=30
            )
            if response.status_code == 200:
                elapsed = (time.perf_counter() - start)
                data = response.json()
                tokens = data.get('usage', {}).get('completion_tokens', 30)
                throughput = tokens / elapsed
                results['korean_throughputs'].append(throughput)
                print(f"  Run {i+1}/5: {throughput:.2f} tok/s")
        except Exception as e:
            print(f"  Error: {e}")

    return results

def calculate_stats(values):
    """Calculate statistics for a list of values"""
    if not values:
        return {}
    return {
        'mean': statistics.mean(values),
        'min': min(values),
        'max': max(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0,
        'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
    }

def main():
    print("="*80)
    print("🎯 SGLang Final Performance Benchmark")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Test configurations
    configs = [
        (8000, 'Baseline-Triton'),
        (8003, 'Balanced-v2-LOF')
    ]

    all_results = []

    for port, name in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name} (Port {port})")
        print(f"{'='*60}")

        try:
            # Check if accessible
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                results = test_configuration(port, name)
                all_results.append(results)
            else:
                print(f"❌ {name} not healthy")
        except Exception as e:
            print(f"❌ {name} not accessible: {e}")

    # Generate CSV report
    print("\n" + "="*80)
    print("📊 FINAL PERFORMANCE COMPARISON")
    print("="*80)

    csv_data = []

    for r in all_results:
        short_stats = calculate_stats(r['short_latencies'])
        medium_stats = calculate_stats(r['medium_throughputs'])
        ttft_stats = calculate_stats(r['ttfts'])
        korean_stats = calculate_stats(r['korean_throughputs'])

        row = {
            'Configuration': r['name'],
            'Port': r['port'],
            'Short_Avg_Latency_ms': round(short_stats.get('mean', 0), 1),
            'Short_Min_Latency_ms': round(short_stats.get('min', 0), 1),
            'Short_P95_Latency_ms': round(short_stats.get('p95', 0), 1),
            'Medium_Avg_Throughput_tps': round(medium_stats.get('mean', 0), 2),
            'Medium_Min_Throughput_tps': round(medium_stats.get('min', 0), 2),
            'Medium_Max_Throughput_tps': round(medium_stats.get('max', 0), 2),
            'TTFT_Avg_ms': round(ttft_stats.get('mean', 0), 1),
            'TTFT_Min_ms': round(ttft_stats.get('min', 0), 1),
            'TTFT_P95_ms': round(ttft_stats.get('p95', 0), 1),
            'Korean_Avg_Throughput_tps': round(korean_stats.get('mean', 0), 2),
            'Test_Runs_Short': len(r['short_latencies']),
            'Test_Runs_Medium': len(r['medium_throughputs']),
            'Test_Runs_TTFT': len(r['ttfts']),
            'Test_Runs_Korean': len(r['korean_throughputs'])
        }
        csv_data.append(row)

        # Print summary
        print(f"\n{r['name']}:")
        print(f"  📌 Short Response: {row['Short_Avg_Latency_ms']}ms (min: {row['Short_Min_Latency_ms']}ms)")
        print(f"  📈 Throughput: {row['Medium_Avg_Throughput_tps']} tok/s")
        print(f"  ⏱️ TTFT: {row['TTFT_Avg_ms']}ms (min: {row['TTFT_Min_ms']}ms)")
        print(f"  🇰🇷 Korean: {row['Korean_Avg_Throughput_tps']} tok/s")

    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'final_benchmark_{timestamp}.csv'

    if csv_data:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = csv_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"\n✅ Results saved to: {filename}")

        # Calculate improvements
        if len(csv_data) == 2:
            baseline = csv_data[0]
            optimized = csv_data[1]

            print("\n" + "="*80)
            print("🚀 PERFORMANCE IMPROVEMENTS (Balanced-v2 vs Baseline)")
            print("="*80)

            latency_improve = (baseline['Short_Avg_Latency_ms'] - optimized['Short_Avg_Latency_ms']) / baseline['Short_Avg_Latency_ms'] * 100
            throughput_improve = (optimized['Medium_Avg_Throughput_tps'] - baseline['Medium_Avg_Throughput_tps']) / baseline['Medium_Avg_Throughput_tps'] * 100
            ttft_improve = (baseline['TTFT_Avg_ms'] - optimized['TTFT_Avg_ms']) / baseline['TTFT_Avg_ms'] * 100
            korean_improve = (optimized['Korean_Avg_Throughput_tps'] - baseline['Korean_Avg_Throughput_tps']) / baseline['Korean_Avg_Throughput_tps'] * 100

            print(f"  ⚡ Response Latency: {latency_improve:+.1f}% improvement")
            print(f"  📈 Throughput: {throughput_improve:+.1f}% improvement")
            print(f"  ⏱️ TTFT: {ttft_improve:+.1f}% improvement")
            print(f"  🇰🇷 Korean Processing: {korean_improve:+.1f}% improvement")

    # Get GPU stats
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"\n📊 GPU Status: {result.stdout.strip()}")
    except:
        pass

    print("\n" + "="*80)
    print("✅ Benchmark Complete!")
    print("="*80)

if __name__ == '__main__':
    main()