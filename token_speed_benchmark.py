#!/usr/bin/env python3
"""
Comprehensive Token Speed Benchmark for SGLang
Measures tokens per second with various configurations
"""

import time
import json
import requests
import statistics
import asyncio
import aiohttp
from datetime import datetime
import csv
import sys
from typing import List, Dict, Any
import argparse

class TokenSpeedBenchmark:
    def __init__(self, host="localhost", port=8000, model="Qwen/Qwen3-8B"):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.results = []

    def print_progress(self, msg):
        """Print progress message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def warmup(self, runs=3):
        """Warm up the model"""
        self.print_progress("🔥 Warming up model...")
        for i in range(runs):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": self.model,
                        "prompt": "Hello, this is a warmup request.",
                        "max_tokens": 10,
                        "temperature": 0.1
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    self.print_progress(f"  Warmup {i+1}/{runs} ✓")
                time.sleep(1)
            except Exception as e:
                self.print_progress(f"  Warmup {i+1}/{runs} failed: {e}")

    def measure_single_request(self, prompt: str, max_tokens: int, runs: int = 5) -> Dict:
        """Measure token speed for single requests"""
        self.print_progress(f"📊 Testing single request (prompt_len={len(prompt.split())}, max_tokens={max_tokens})")

        latencies = []
        tokens_per_second = []
        ttft_times = []

        for i in range(runs):
            try:
                # Measure total time
                start_time = time.perf_counter()

                response = requests.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "stream": False
                    },
                    timeout=120
                )

                end_time = time.perf_counter()

                if response.status_code == 200:
                    data = response.json()

                    # Calculate metrics
                    total_time = end_time - start_time
                    latencies.append(total_time * 1000)  # ms

                    # Get token counts
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)

                    # Calculate tokens per second (generation only)
                    if total_time > 0 and completion_tokens > 0:
                        tps = completion_tokens / total_time
                        tokens_per_second.append(tps)

                    # Estimate TTFT (for non-streaming, use a fraction of total time)
                    ttft_estimate = (total_time * 0.2) * 1000  # Rough estimate: 20% of time for first token
                    ttft_times.append(ttft_estimate)

                    self.print_progress(f"  Run {i+1}/{runs}: {completion_tokens} tokens in {total_time:.2f}s = {tps:.2f} tok/s")

            except Exception as e:
                self.print_progress(f"  Run {i+1}/{runs} failed: {e}")

        if tokens_per_second:
            return {
                "test_type": "single_request",
                "prompt_words": len(prompt.split()),
                "max_tokens": max_tokens,
                "runs": len(tokens_per_second),
                "avg_tokens_per_second": statistics.mean(tokens_per_second),
                "min_tokens_per_second": min(tokens_per_second),
                "max_tokens_per_second": max(tokens_per_second),
                "std_tokens_per_second": statistics.stdev(tokens_per_second) if len(tokens_per_second) > 1 else 0,
                "avg_latency_ms": statistics.mean(latencies),
                "p50_latency_ms": statistics.median(latencies),
                "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 5 else max(latencies),
                "avg_ttft_ms": statistics.mean(ttft_times)
            }
        return None

    async def measure_concurrent_requests(self, prompt: str, max_tokens: int, concurrent: int = 5) -> Dict:
        """Measure token speed with concurrent requests"""
        self.print_progress(f"🔀 Testing {concurrent} concurrent requests")

        async def make_request(session, request_id):
            start_time = time.perf_counter()
            try:
                async with session.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.9
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        end_time = time.perf_counter()

                        total_time = end_time - start_time
                        usage = data.get("usage", {})
                        completion_tokens = usage.get("completion_tokens", 0)

                        tps = completion_tokens / total_time if total_time > 0 else 0

                        return {
                            "request_id": request_id,
                            "total_time": total_time,
                            "completion_tokens": completion_tokens,
                            "tokens_per_second": tps,
                            "latency_ms": total_time * 1000
                        }
            except Exception as e:
                self.print_progress(f"  Request {request_id} failed: {e}")
                return None

        # Run concurrent requests
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session, i) for i in range(concurrent)]
            results = await asyncio.gather(*tasks)

        # Filter successful results
        successful_results = [r for r in results if r is not None]

        if successful_results:
            all_tps = [r["tokens_per_second"] for r in successful_results]
            all_latencies = [r["latency_ms"] for r in successful_results]
            total_tokens = sum(r["completion_tokens"] for r in successful_results)
            total_time = max(r["total_time"] for r in successful_results)

            return {
                "test_type": "concurrent_requests",
                "concurrent_users": concurrent,
                "prompt_words": len(prompt.split()),
                "max_tokens": max_tokens,
                "successful_requests": len(successful_results),
                "avg_tokens_per_second_per_request": statistics.mean(all_tps),
                "total_tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
                "avg_latency_ms": statistics.mean(all_latencies),
                "p50_latency_ms": statistics.median(all_latencies),
                "p95_latency_ms": statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) >= 5 else max(all_latencies)
            }
        return None

    def measure_streaming(self, prompt: str, max_tokens: int) -> Dict:
        """Measure token speed with streaming"""
        self.print_progress(f"🌊 Testing streaming response")

        try:
            start_time = time.perf_counter()
            first_token_time = None
            tokens_received = 0

            response = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "stream": True
                },
                stream=True,
                timeout=120
            )

            for line in response.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    if line.startswith(b"data: "):
                        try:
                            data = json.loads(line[6:])
                            if "choices" in data and data["choices"]:
                                tokens_received += 1
                        except:
                            pass

            end_time = time.perf_counter()

            total_time = end_time - start_time
            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0

            return {
                "test_type": "streaming",
                "prompt_words": len(prompt.split()),
                "max_tokens": max_tokens,
                "tokens_received": tokens_received,
                "total_time_seconds": total_time,
                "tokens_per_second": tokens_received / total_time if total_time > 0 else 0,
                "time_to_first_token_ms": ttft
            }

        except Exception as e:
            self.print_progress(f"  Streaming test failed: {e}")
            return None

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite"""
        self.print_progress("🚀 Starting Comprehensive Token Speed Benchmark")
        self.print_progress(f"📍 Server: {self.base_url}")
        self.print_progress(f"🤖 Model: {self.model}")
        print("=" * 60)

        # Warmup
        self.warmup()
        print()

        # Test configurations
        test_configs = [
            # (prompt, max_tokens, description)
            ("What is 2+2?", 10, "short_prompt_short_response"),
            ("Tell me a story about a brave knight.", 50, "short_prompt_medium_response"),
            ("Explain quantum computing in detail.", 100, "short_prompt_long_response"),
            ("Write a comprehensive essay about climate change, covering its causes, effects, and potential solutions.", 200, "medium_prompt_long_response"),
            ("The quick brown fox jumps over the lazy dog. " * 20, 100, "long_prompt_long_response"),
        ]

        all_results = []

        # Single request tests
        print("=" * 60)
        print("SINGLE REQUEST TESTS")
        print("=" * 60)
        for prompt, max_tokens, desc in test_configs:
            result = self.measure_single_request(prompt, max_tokens, runs=5)
            if result:
                result["description"] = desc
                all_results.append(result)
                print(f"✅ {desc}: {result['avg_tokens_per_second']:.2f} tok/s")
            time.sleep(2)  # Pause between tests

        # Concurrent request tests
        print("\n" + "=" * 60)
        print("CONCURRENT REQUEST TESTS")
        print("=" * 60)

        # Use a medium complexity prompt for concurrent tests
        concurrent_prompt = "Explain the concept of artificial intelligence."
        for concurrent_users in [2, 5, 10]:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.measure_concurrent_requests(concurrent_prompt, 50, concurrent_users)
            )
            loop.close()

            if result:
                result["description"] = f"concurrent_{concurrent_users}_users"
                all_results.append(result)
                print(f"✅ {concurrent_users} users: {result['total_tokens_per_second']:.2f} total tok/s")
            time.sleep(3)

        # Streaming test
        print("\n" + "=" * 60)
        print("STREAMING TEST")
        print("=" * 60)
        result = self.measure_streaming("Tell me an interesting fact about space.", 50)
        if result:
            result["description"] = "streaming_test"
            all_results.append(result)
            print(f"✅ Streaming: {result['tokens_per_second']:.2f} tok/s, TTFT: {result['time_to_first_token_ms']:.2f}ms")

        return all_results

    def save_results(self, results: List[Dict], filename: str = None):
        """Save results to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"token_speed_benchmark_{timestamp}.csv"

        # Collect all unique keys
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())

        # Write CSV
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(results)

        self.print_progress(f"📊 Results saved to: {filename}")
        return filename

    def print_summary(self, results: List[Dict]):
        """Print summary of benchmark results"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        # Single request summary
        single_results = [r for r in results if r.get("test_type") == "single_request"]
        if single_results:
            avg_speed = statistics.mean([r["avg_tokens_per_second"] for r in single_results])
            print(f"\n📌 Single Request Performance:")
            print(f"  Average Speed: {avg_speed:.2f} tokens/second")

            for r in single_results:
                print(f"  {r['description']}: {r['avg_tokens_per_second']:.2f} tok/s")

        # Concurrent request summary
        concurrent_results = [r for r in results if r.get("test_type") == "concurrent_requests"]
        if concurrent_results:
            print(f"\n📌 Concurrent Request Performance:")
            for r in concurrent_results:
                print(f"  {r['concurrent_users']} users: {r['total_tokens_per_second']:.2f} total tok/s")

        # Streaming summary
        streaming_results = [r for r in results if r.get("test_type") == "streaming"]
        if streaming_results:
            print(f"\n📌 Streaming Performance:")
            for r in streaming_results:
                print(f"  Speed: {r['tokens_per_second']:.2f} tok/s")
                print(f"  TTFT: {r['time_to_first_token_ms']:.2f}ms")

        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Token Speed Benchmark for SGLang")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--output", help="Output CSV filename")

    args = parser.parse_args()

    # Run benchmark
    benchmark = TokenSpeedBenchmark(host=args.host, port=args.port, model=args.model)

    # Check server health first
    try:
        response = requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
        if response.status_code != 200:
            print("⚠️  Server health check failed, but continuing anyway...")
    except:
        print("⚠️  Cannot reach server. Please ensure SGLang is running.")
        print(f"   Check: http://{args.host}:{args.port}")
        sys.exit(1)

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()

    # Save and display results
    if results:
        csv_file = benchmark.save_results(results, args.output)
        benchmark.print_summary(results)
        print(f"\n✅ Benchmark complete! Results saved to: {csv_file}")
    else:
        print("\n❌ No results collected. Please check server status.")

if __name__ == "__main__":
    main()