#!/usr/bin/env python3
"""
Concurrent Stress Test - Multiple simultaneous requests for token generation
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
import statistics

async def make_concurrent_request(session, request_id, prompt, max_tokens, url):
    """Make a single async request"""
    start_time = time.perf_counter()

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                end_time = time.perf_counter()

                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                total_time = end_time - start_time
                tokens_per_second = completion_tokens / total_time if total_time > 0 else 0

                return {
                    "request_id": request_id,
                    "success": True,
                    "tokens": completion_tokens,
                    "time": total_time,
                    "speed": tokens_per_second,
                    "start": start_time,
                    "end": end_time
                }
            else:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": f"Status {response.status}"
                }
    except Exception as e:
        return {
            "request_id": request_id,
            "success": False,
            "error": str(e)
        }

async def run_concurrent_test(num_concurrent, tokens_per_request):
    """Run concurrent test with specified number of simultaneous requests"""

    url = "http://localhost:8000/v1/completions"

    # Different prompts for variety
    prompts = [
        "Write a detailed story about artificial intelligence and the future of humanity:",
        "영어, 한국어, 일본어, 중국어를 번갈아가면서 시를 작성해줘:",
        "Explain quantum computing with detailed technical examples:",
        "Create a comprehensive business plan for a tech startup:",
        "Write a mystery novel chapter with suspense and plot twists:",
        "Describe the process of machine learning in great detail:",
        "작은 마을에서 일어난 미스터리한 사건에 대한 이야기를 써줘:",
        "技術革新が社会に与える影響について詳しく説明してください：",
        "详细解释区块链技术的工作原理和应用场景：",
        "Write a scientific paper abstract about climate change:",
    ]

    print(f"\n{'='*60}")
    print(f"🔥 Testing {num_concurrent} Concurrent Requests")
    print(f"📝 {tokens_per_request} tokens per request")
    print(f"{'='*60}")

    # Prepare requests
    tasks = []
    async with aiohttp.ClientSession() as session:
        for i in range(num_concurrent):
            prompt = prompts[i % len(prompts)]
            task = make_concurrent_request(
                session,
                i+1,
                prompt,
                tokens_per_request,
                url
            )
            tasks.append(task)

        # Execute all requests simultaneously
        print(f"⚡ Launching {num_concurrent} simultaneous requests...")
        overall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        overall_end = time.perf_counter()
        overall_time = overall_end - overall_start

    # Analyze results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    if successful:
        total_tokens = sum(r["tokens"] for r in successful)
        individual_speeds = [r["speed"] for r in successful]
        individual_times = [r["time"] for r in successful]

        # Calculate metrics
        avg_individual_speed = statistics.mean(individual_speeds)
        min_speed = min(individual_speeds)
        max_speed = max(individual_speeds)
        avg_response_time = statistics.mean(individual_times)

        # Overall throughput
        overall_throughput = total_tokens / overall_time if overall_time > 0 else 0

        print(f"\n📊 Results:")
        print(f"  ✅ Successful: {len(successful)}/{num_concurrent}")
        print(f"  ❌ Failed: {len(failed)}/{num_concurrent}")

        print(f"\n⏱️  Timing:")
        print(f"  Overall time: {overall_time:.2f} seconds")
        print(f"  Avg response time: {avg_response_time:.2f} seconds")
        print(f"  Min response time: {min(individual_times):.2f} seconds")
        print(f"  Max response time: {max(individual_times):.2f} seconds")

        print(f"\n🚀 Token Generation Speed:")
        print(f"  Total tokens generated: {total_tokens}")
        print(f"  Overall throughput: {overall_throughput:.2f} tok/s")
        print(f"  Average individual speed: {avg_individual_speed:.2f} tok/s")
        print(f"  Min individual speed: {min_speed:.2f} tok/s")
        print(f"  Max individual speed: {max_speed:.2f} tok/s")

        print(f"\n📈 Performance Metrics:")
        print(f"  Tokens per minute: {overall_throughput * 60:.0f}")
        print(f"  Requests per second: {len(successful) / overall_time:.2f}")
        print(f"  Efficiency: {(avg_individual_speed * num_concurrent) / overall_throughput:.1%}")

        # Show failed requests if any
        if failed:
            print(f"\n⚠️  Failed Requests:")
            for f in failed:
                print(f"    Request {f['request_id']}: {f.get('error', 'Unknown error')}")

        return {
            "concurrent": num_concurrent,
            "tokens_per_request": tokens_per_request,
            "total_tokens": total_tokens,
            "overall_time": overall_time,
            "throughput": overall_throughput,
            "success_rate": len(successful) / num_concurrent
        }
    else:
        print(f"\n❌ All requests failed!")
        return None

async def main():
    """Run multiple concurrent test scenarios"""

    print("🚀 Concurrent Token Generation Stress Test")
    print("=" * 60)
    print("Testing server capacity with simultaneous requests")
    print("=" * 60)

    # Test scenarios: (concurrent_requests, tokens_per_request)
    test_scenarios = [
        (1, 100),    # Baseline: single request
        (2, 100),    # 2 concurrent
        (5, 100),    # 5 concurrent
        (10, 100),   # 10 concurrent
        (20, 100),   # 20 concurrent
        (5, 500),    # 5 concurrent with more tokens
        (10, 500),   # 10 concurrent with more tokens
        (5, 1000),   # 5 concurrent with heavy load
    ]

    results = []

    for concurrent, tokens in test_scenarios:
        result = await run_concurrent_test(concurrent, tokens)
        if result:
            results.append(result)

        # Brief pause between tests
        await asyncio.sleep(2)

    # Final summary
    print("\n" + "="*60)
    print("🏁 FINAL SUMMARY")
    print("="*60)

    if results:
        print("\n📊 Throughput by Concurrent Users:")
        for r in results:
            print(f"  {r['concurrent']:2d} users × {r['tokens_per_request']:4d} tokens: "
                  f"{r['throughput']:7.2f} tok/s "
                  f"({r['success_rate']:.0%} success)")

        # Find best configuration
        best = max(results, key=lambda x: x['throughput'])
        print(f"\n🏆 Best Performance:")
        print(f"  Configuration: {best['concurrent']} concurrent requests")
        print(f"  Throughput: {best['throughput']:.2f} tok/s")
        print(f"  Total tokens: {best['total_tokens']} in {best['overall_time']:.2f}s")

        # Performance rating
        max_throughput = max(r['throughput'] for r in results)
        print(f"\n🎯 Server Capacity Rating:")
        if max_throughput > 500:
            print("  ⭐⭐⭐⭐⭐ ENTERPRISE GRADE (>500 tok/s)")
        elif max_throughput > 300:
            print("  ⭐⭐⭐⭐ PRODUCTION READY (300-500 tok/s)")
        elif max_throughput > 150:
            print("  ⭐⭐⭐ GOOD CAPACITY (150-300 tok/s)")
        elif max_throughput > 75:
            print("  ⭐⭐ MODERATE (75-150 tok/s)")
        else:
            print("  ⭐ LIMITED (<75 tok/s)")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"concurrent_test_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "scenarios": results
        }, f, indent=2)

    print(f"\n💾 Report saved to: {report_file}")

if __name__ == "__main__":
    # Check server first
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print("✅ Server is ready\n")

        # Run async tests
        asyncio.run(main())
    except:
        print("❌ Server is not responding. Please check if SGLang is running on port 8000")
        exit(1)