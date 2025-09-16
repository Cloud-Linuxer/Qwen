#!/usr/bin/env python3
"""
SGLang Comprehensive Performance Benchmark
Generates detailed CSV output with all metrics
"""

import time
import json
import requests
import csv
import statistics
from datetime import datetime
import subprocess
import sys

class SGLangBenchmark:
    def __init__(self, port, name, config_desc):
        self.port = port
        self.name = name
        self.config_desc = config_desc
        self.base_url = f"http://localhost:{port}"
        self.results = []

    def warmup(self):
        """Warm up the model with a few requests"""
        print(f"🔥 Warming up {self.name}...")
        for _ in range(3):
            try:
                requests.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": "Qwen/Qwen3-32B-AWQ",
                        "prompt": "Hello",
                        "max_tokens": 5,
                        "temperature": 0.1
                    },
                    timeout=10
                )
                time.sleep(1)
            except:
                pass

    def test_latency(self, prompt, max_tokens, num_runs=10):
        """Test response latency"""
        latencies = []
        tokens_generated = []

        for i in range(num_runs):
            start = time.perf_counter()
            try:
                response = requests.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": "Qwen/Qwen3-32B-AWQ",
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.1
                    },
                    timeout=30
                )
                end = time.perf_counter()

                if response.status_code == 200:
                    data = response.json()
                    latency = (end - start) * 1000  # Convert to ms
                    latencies.append(latency)
                    tokens = data.get('usage', {}).get('completion_tokens', max_tokens)
                    tokens_generated.append(tokens)
            except Exception as e:
                print(f"  ❌ Error in run {i+1}: {e}")

        if latencies:
            return {
                'avg_latency': statistics.mean(latencies),
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'std_latency': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'p50_latency': statistics.median(latencies),
                'p95_latency': statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                'avg_tokens': statistics.mean(tokens_generated),
                'throughput': statistics.mean([t / (l / 1000) for t, l in zip(tokens_generated, latencies)])
            }
        return None

    def test_ttft(self, prompt="Tell me a story:", max_tokens=20):
        """Test Time to First Token with streaming"""
        try:
            start = time.perf_counter()
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": "Qwen/Qwen3-32B-AWQ",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.5,
                    "stream": True
                },
                stream=True,
                timeout=10
            )

            # Get first chunk
            for line in response.iter_lines():
                if line:
                    ttft = (time.perf_counter() - start) * 1000
                    return ttft
        except Exception as e:
            print(f"  ❌ TTFT error: {e}")
        return None

    def test_concurrent(self, num_requests=5, max_tokens=30):
        """Test concurrent request handling"""
        import concurrent.futures

        def make_request(i):
            start = time.perf_counter()
            try:
                response = requests.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": "Qwen/Qwen3-32B-AWQ",
                        "prompt": f"Request {i}: Explain quantum computing in simple terms:",
                        "max_tokens": max_tokens,
                        "temperature": 0.3
                    },
                    timeout=60
                )
                end = time.perf_counter()
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'latency': (end - start) * 1000,
                        'tokens': data.get('usage', {}).get('completion_tokens', 0)
                    }
            except:
                pass
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            start_time = time.perf_counter()
            results = list(executor.map(make_request, range(num_requests)))
            total_time = (time.perf_counter() - start_time) * 1000

        successful = [r for r in results if r]
        if successful:
            total_tokens = sum(r['tokens'] for r in successful)
            avg_latency = statistics.mean([r['latency'] for r in successful])
            return {
                'total_time': total_time,
                'avg_latency': avg_latency,
                'total_tokens': total_tokens,
                'throughput': total_tokens / (total_time / 1000) if total_time > 0 else 0,
                'success_rate': len(successful) / num_requests * 100
            }
        return None

    def get_gpu_metrics(self):
        """Get current GPU metrics"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'gpu_memory_used': float(values[0]),
                    'gpu_memory_total': float(values[1]),
                    'gpu_temp': float(values[2]),
                    'gpu_power': float(values[3])
                }
        except:
            pass
        return {}

    def run_benchmark(self):
        """Run complete benchmark suite"""
        print(f"\n{'='*70}")
        print(f"🎯 Benchmarking: {self.name}")
        print(f"   Config: {self.config_desc}")
        print(f"   Port: {self.port}")
        print(f"{'='*70}")

        self.warmup()

        # Test 1: Short response (10 tokens)
        print("\n📊 Test 1: Short Response (10 tokens)")
        short_result = self.test_latency("The capital of France is", 10, num_runs=20)

        # Test 2: Medium response (50 tokens)
        print("📊 Test 2: Medium Response (50 tokens)")
        medium_result = self.test_latency("Write about artificial intelligence:", 50, num_runs=10)

        # Test 3: Long response (100 tokens)
        print("📊 Test 3: Long Response (100 tokens)")
        long_result = self.test_latency("Explain the theory of relativity in detail:", 100, num_runs=5)

        # Test 4: Very long response (200 tokens)
        print("📊 Test 4: Very Long Response (200 tokens)")
        vlong_result = self.test_latency("Write a detailed story about space exploration:", 200, num_runs=3)

        # Test 5: TTFT
        print("📊 Test 5: Time to First Token")
        ttft_results = []
        for _ in range(5):
            ttft = self.test_ttft()
            if ttft:
                ttft_results.append(ttft)
        avg_ttft = statistics.mean(ttft_results) if ttft_results else None

        # Test 6: Concurrent requests
        print("📊 Test 6: Concurrent Requests (5 parallel)")
        concurrent_result = self.test_concurrent(5, 30)

        # Test 7: Korean language
        print("📊 Test 7: Korean Language (30 tokens)")
        korean_result = self.test_latency("인공지능의 장점과 단점을 설명해주세요:", 30, num_runs=5)

        # Get GPU metrics
        gpu_metrics = self.get_gpu_metrics()

        # Compile results
        result = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.name,
            'port': self.port,
            'config_details': self.config_desc,

            # Short response metrics
            'short_avg_latency_ms': short_result['avg_latency'] if short_result else None,
            'short_min_latency_ms': short_result['min_latency'] if short_result else None,
            'short_max_latency_ms': short_result['max_latency'] if short_result else None,
            'short_p50_latency_ms': short_result['p50_latency'] if short_result else None,
            'short_p95_latency_ms': short_result['p95_latency'] if short_result else None,
            'short_throughput_tps': short_result['throughput'] if short_result else None,

            # Medium response metrics
            'medium_avg_latency_ms': medium_result['avg_latency'] if medium_result else None,
            'medium_throughput_tps': medium_result['throughput'] if medium_result else None,

            # Long response metrics
            'long_avg_latency_ms': long_result['avg_latency'] if long_result else None,
            'long_throughput_tps': long_result['throughput'] if long_result else None,

            # Very long response metrics
            'vlong_avg_latency_ms': vlong_result['avg_latency'] if vlong_result else None,
            'vlong_throughput_tps': vlong_result['throughput'] if vlong_result else None,

            # TTFT
            'avg_ttft_ms': avg_ttft,

            # Concurrent metrics
            'concurrent_total_time_ms': concurrent_result['total_time'] if concurrent_result else None,
            'concurrent_avg_latency_ms': concurrent_result['avg_latency'] if concurrent_result else None,
            'concurrent_throughput_tps': concurrent_result['throughput'] if concurrent_result else None,
            'concurrent_success_rate': concurrent_result['success_rate'] if concurrent_result else None,

            # Korean language
            'korean_avg_latency_ms': korean_result['avg_latency'] if korean_result else None,
            'korean_throughput_tps': korean_result['throughput'] if korean_result else None,

            # GPU metrics
            **gpu_metrics
        }

        # Print summary
        print(f"\n{'='*70}")
        print(f"✅ Benchmark Complete: {self.name}")
        if short_result:
            print(f"   Short Response: {short_result['avg_latency']:.0f}ms")
        if medium_result:
            print(f"   Throughput (50 tok): {medium_result['throughput']:.2f} tok/s")
        if avg_ttft:
            print(f"   TTFT: {avg_ttft:.0f}ms")
        print(f"{'='*70}\n")

        return result

def save_to_csv(results, filename='benchmark_results.csv'):
    """Save results to CSV file"""
    if not results:
        return

    fieldnames = results[0].keys()

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"📝 Results saved to {filename}")

def main():
    configurations = [
        {
            'port': 8000,
            'name': 'Baseline-Triton',
            'desc': 'Initial optimization with Triton attention'
        },
        {
            'port': 8001,
            'name': 'Ultra-Optimized',
            'desc': 'Torch Compile + LPM scheduling + 3 decode steps'
        },
        {
            'port': 8003,
            'name': 'Balanced-v2',
            'desc': 'LOF scheduling + Torch Compile + 2 decode steps'
        }
    ]

    all_results = []

    print("🚀 SGLang Comprehensive Performance Benchmark")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️ Testing {len(configurations)} configurations")

    for config in configurations:
        # Check if port is accessible
        try:
            response = requests.get(f"http://localhost:{config['port']}/health", timeout=2)
            if response.status_code == 200:
                benchmark = SGLangBenchmark(config['port'], config['name'], config['desc'])
                result = benchmark.run_benchmark()
                if result:
                    all_results.append(result)
            else:
                print(f"⚠️ {config['name']} on port {config['port']} not healthy")
        except:
            print(f"❌ {config['name']} on port {config['port']} not accessible")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'sglang_benchmark_{timestamp}.csv'
    save_to_csv(all_results, filename)

    # Print comparison table
    if all_results:
        print("\n" + "="*80)
        print("📊 PERFORMANCE COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Configuration':<20} {'Latency(ms)':<15} {'Throughput':<15} {'TTFT(ms)':<10}")
        print("-"*80)

        for r in all_results:
            print(f"{r['configuration']:<20} "
                  f"{r.get('short_avg_latency_ms', 0):<15.0f} "
                  f"{r.get('medium_throughput_tps', 0):<15.2f} "
                  f"{r.get('avg_ttft_ms', 0):<10.0f}")
        print("="*80)

if __name__ == '__main__':
    main()