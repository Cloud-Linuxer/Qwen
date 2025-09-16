#!/usr/bin/env python3
"""
Heavy Generation Test - Maximum token generation speed test
"""

import time
import requests
import json
from datetime import datetime

def heavy_generation_test(port=8000):
    """Test with multiple requests to generate thousands of tokens"""

    url = f"http://localhost:{port}/v1/completions"

    print("🔥 Heavy Token Generation Test")
    print("=" * 60)
    print("Testing maximum sustained generation speed")
    print("=" * 60)
    print()

    # Different test scenarios
    test_scenarios = [
        {
            "name": "Short burst (100 tokens)",
            "prompt": "Write a detailed explanation about artificial intelligence:",
            "max_tokens": 100,
            "runs": 5
        },
        {
            "name": "Medium generation (500 tokens)",
            "prompt": "영어, 한국어, 일본어, 중국어를 번갈아가면서 사계절에 대한 긴 시를 작성해줘:\n\n[English]\nSpring arrives with colors bright,\n\n[한국어]\n여름의 열기가 대지를 덮고,\n\n[日本語]\n秋の風が涼しく吹き、\n\n[中文]\n冬天的雪花飘落，\n\n계속:",
            "max_tokens": 500,
            "runs": 3
        },
        {
            "name": "Large generation (2000 tokens)",
            "prompt": "Write an extremely detailed technical documentation about implementing a distributed system with microservices architecture. Include code examples, diagrams descriptions, and best practices. Be very thorough and comprehensive:",
            "max_tokens": 2000,
            "runs": 2
        },
        {
            "name": "Maximum generation (4000 tokens)",
            "prompt": "Create a complete novel chapter with multiple characters, detailed descriptions, dialogues, and plot development. Make it engaging and detailed:",
            "max_tokens": 4000,
            "runs": 1
        }
    ]

    total_tokens_generated = 0
    total_time_spent = 0
    results = []

    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"📋 Test: {scenario['name']}")
        print(f"🎯 Target: {scenario['max_tokens']} tokens × {scenario['runs']} runs")
        print(f"{'='*60}")

        scenario_tokens = 0
        scenario_time = 0
        speeds = []

        for run in range(scenario['runs']):
            print(f"\n  Run {run+1}/{scenario['runs']}:")
            start_time = time.perf_counter()

            try:
                response = requests.post(
                    url,
                    json={
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "prompt": scenario['prompt'],
                        "max_tokens": scenario['max_tokens'],
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                    timeout=300
                )

                end_time = time.perf_counter()
                elapsed = end_time - start_time

                if response.status_code == 200:
                    data = response.json()
                    usage = data.get("usage", {})
                    completion_tokens = usage.get("completion_tokens", 0)

                    speed = completion_tokens / elapsed if elapsed > 0 else 0
                    speeds.append(speed)

                    scenario_tokens += completion_tokens
                    scenario_time += elapsed

                    print(f"    ✅ Generated: {completion_tokens} tokens")
                    print(f"    ⏱️  Time: {elapsed:.2f}s")
                    print(f"    🚀 Speed: {speed:.2f} tok/s")
                else:
                    print(f"    ❌ Failed: {response.status_code}")

                time.sleep(1)  # Brief pause between runs

            except Exception as e:
                print(f"    ❌ Error: {e}")

        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            print(f"\n  📊 Scenario Summary:")
            print(f"    Total tokens: {scenario_tokens}")
            print(f"    Total time: {scenario_time:.2f}s")
            print(f"    Average speed: {avg_speed:.2f} tok/s")
            print(f"    Min speed: {min(speeds):.2f} tok/s")
            print(f"    Max speed: {max(speeds):.2f} tok/s")

            results.append({
                "scenario": scenario['name'],
                "tokens": scenario_tokens,
                "time": scenario_time,
                "avg_speed": avg_speed,
                "runs": len(speeds)
            })

            total_tokens_generated += scenario_tokens
            total_time_spent += scenario_time

    # Final summary
    print("\n" + "="*60)
    print("🏁 FINAL RESULTS")
    print("="*60)
    print(f"\n📊 Overall Performance:")
    print(f"  Total tokens generated: {total_tokens_generated}")
    print(f"  Total time: {total_time_spent:.2f} seconds")
    print(f"  Overall speed: {total_tokens_generated/total_time_spent:.2f} tok/s")
    print(f"  Throughput: {total_tokens_generated/total_time_spent*60:.0f} tokens/minute")

    print(f"\n📈 Per-Scenario Results:")
    for r in results:
        print(f"  {r['scenario']}:")
        print(f"    Average: {r['avg_speed']:.2f} tok/s over {r['runs']} runs")

    # Performance rating
    overall_speed = total_tokens_generated/total_time_spent if total_time_spent > 0 else 0
    print(f"\n🎯 Performance Rating:")
    if overall_speed > 100:
        print("  ⭐⭐⭐⭐⭐ EXCEPTIONAL (>100 tok/s)")
    elif overall_speed > 75:
        print("  ⭐⭐⭐⭐ EXCELLENT (75-100 tok/s)")
    elif overall_speed > 50:
        print("  ⭐⭐⭐ GOOD (50-75 tok/s)")
    elif overall_speed > 25:
        print("  ⭐⭐ MODERATE (25-50 tok/s)")
    else:
        print("  ⭐ NEEDS OPTIMIZATION (<25 tok/s)")

    # Save detailed report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"heavy_test_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "total_tokens": total_tokens_generated,
            "total_time": total_time_spent,
            "overall_speed": overall_speed,
            "scenarios": results
        }, f, indent=2)

    print(f"\n💾 Detailed report saved to: {report_file}")

def main():
    print("🚀 Starting Heavy Generation Test")
    print("This will test sustained token generation across multiple scenarios\n")

    # Check server
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print("✅ Server is ready\n")
    except:
        print("⚠️  Server may not be ready, but continuing...\n")

    heavy_generation_test()

if __name__ == "__main__":
    main()