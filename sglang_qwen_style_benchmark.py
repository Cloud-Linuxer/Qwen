#!/usr/bin/env python3
"""
SGLang Performance Test - Qwen Style Benchmark
Based on qwen_performance_test.py format for direct comparison
"""

import json
import time
import requests
import statistics
import csv
from datetime import datetime
from typing import List, Dict, Any
import concurrent.futures
import numpy as np

class SGLangPerformanceTester:
    def __init__(self, base_url: str, config_name: str):
        self.base_url = base_url
        self.config_name = config_name
        self.model_id = "Qwen/Qwen3-32B-AWQ"
        self.results = []

    def test_single_request(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        """Test a single request and measure latency"""
        url = f"{self.base_url}/v1/completions"

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        start_time = time.time()
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            end_time = time.time()

            result = response.json()

            # Calculate metrics
            latency = end_time - start_time
            tokens_generated = result.get('usage', {}).get('completion_tokens', 0)
            tokens_per_second = tokens_generated / latency if latency > 0 else 0

            return {
                "success": True,
                "latency": latency,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_per_second,
                "response_length": len(result['choices'][0]['text'])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latency": time.time() - start_time
            }

    def test_throughput(self, num_requests: int = 10, prompt: str = "Hello, how are you?") -> Dict[str, Any]:
        """Test throughput with concurrent requests"""
        print(f"  동시 사용자 {num_requests}명 테스트...")

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(self.test_single_request, prompt, 50) for _ in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        end_time = time.time()

        successful = [r for r in results if r.get("success")]
        total_time = end_time - start_time

        return {
            "test_type": "throughput",
            "total_requests": num_requests,
            "successful_requests": len(successful),
            "failed_requests": num_requests - len(successful),
            "total_time": total_time,
            "requests_per_second": num_requests / total_time if total_time > 0 else 0,
            "avg_latency": statistics.mean([r["latency"] for r in successful]) if successful else 0,
            "min_latency": min([r["latency"] for r in successful]) if successful else 0,
            "max_latency": max([r["latency"] for r in successful]) if successful else 0
        }

    def test_token_generation_speed(self, token_counts: List[int] = [10, 50, 100, 200, 500]) -> List[Dict[str, Any]]:
        """Test generation speed for different token counts"""
        print("  토큰 생성 속도 테스트...")
        results = []

        prompt = "Write a detailed explanation about artificial intelligence"

        for max_tokens in token_counts:
            print(f"    {max_tokens} 토큰...")
            result = self.test_single_request(prompt, max_tokens=max_tokens, temperature=0.7)
            if result["success"]:
                results.append({
                    "test_type": "token_generation",
                    "max_tokens": max_tokens,
                    "actual_tokens": result["tokens_generated"],
                    "latency": result["latency"],
                    "tokens_per_second": result["tokens_per_second"]
                })

        return results

    def test_prompt_length_impact(self, base_prompt: str = "Explain", token_additions: List[str] = ["", "in detail", "with examples", "comprehensively covering all aspects"]) -> List[Dict[str, Any]]:
        """Test impact of prompt length on response time"""
        print("  프롬프트 길이 영향 테스트...")
        results = []

        for addition in token_additions:
            prompt = f"{base_prompt} {addition} quantum computing"
            prompt_tokens = len(prompt.split())

            result = self.test_single_request(prompt, max_tokens=100)
            if result["success"]:
                results.append({
                    "test_type": "prompt_length",
                    "prompt_tokens": prompt_tokens,
                    "latency": result["latency"],
                    "tokens_per_second": result["tokens_per_second"]
                })

        return results

    def test_temperature_impact(self, temperatures: List[float] = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]) -> List[Dict[str, Any]]:
        """Test impact of temperature on generation speed"""
        print("  온도 설정 영향 테스트...")
        results = []

        prompt = "Generate a creative story about"

        for temp in temperatures:
            result = self.test_single_request(prompt, max_tokens=100, temperature=temp)
            if result["success"]:
                results.append({
                    "test_type": "temperature",
                    "temperature": temp,
                    "latency": result["latency"],
                    "tokens_per_second": result["tokens_per_second"]
                })

        return results

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and compile results"""
        print(f"\n🎯 {self.config_name} 테스트 시작...")

        # Warmup
        print("  워밍업...")
        for _ in range(3):
            self.test_single_request("Hello", max_tokens=5)
            time.sleep(0.5)

        all_results = {
            "configuration": self.config_name,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_id
        }

        # 1. Token generation tests
        token_results = self.test_token_generation_speed([10, 50, 100, 200, 500])
        all_results["token_generation"] = token_results

        # 2. Prompt length tests
        prompt_results = self.test_prompt_length_impact()
        all_results["prompt_length"] = prompt_results

        # 3. Temperature tests
        temp_results = self.test_temperature_impact()
        all_results["temperature"] = temp_results

        # 4. Throughput tests
        throughput_results = []
        for num_users in [1, 2, 5, 10, 20]:
            result = self.test_throughput(num_users)
            throughput_results.append(result)
        all_results["throughput"] = throughput_results

        return all_results

def save_results_to_csv(results: List[Dict], filename: str):
    """Save results in Qwen format CSV"""

    rows = []

    for config_result in results:
        config_name = config_result["configuration"]

        # Token generation performance
        if "token_generation" in config_result:
            for test in config_result["token_generation"]:
                rows.append([
                    config_name,
                    "토큰 생성 성능",
                    f"{test['max_tokens']} 토큰 생성 지연시간",
                    round(test["latency"], 2),
                    "seconds",
                    f"{test['actual_tokens']} 토큰 실제 생성"
                ])
                rows.append([
                    config_name,
                    "토큰 생성 성능",
                    f"{test['max_tokens']} 토큰 생성 속도",
                    round(test["tokens_per_second"], 2),
                    "tokens/sec",
                    ""
                ])

        # Prompt length impact
        if "prompt_length" in config_result:
            for test in config_result["prompt_length"]:
                rows.append([
                    config_name,
                    "프롬프트 길이 영향",
                    f"프롬프트 {test['prompt_tokens']} 토큰",
                    round(test["latency"], 2),
                    "seconds",
                    ""
                ])

        # Temperature impact
        if "temperature" in config_result:
            for test in config_result["temperature"]:
                rows.append([
                    config_name,
                    "온도 설정 영향",
                    f"온도 {test['temperature']} 지연시간",
                    round(test["latency"], 2),
                    "seconds",
                    ""
                ])

        # Throughput tests
        if "throughput" in config_result:
            for test in config_result["throughput"]:
                rows.append([
                    config_name,
                    "동시 사용자 처리",
                    f"{test['total_requests']}명 처리량",
                    round(test["requests_per_second"], 2),
                    "req/sec",
                    f"{test['total_requests']}명 동시"
                ])
                rows.append([
                    config_name,
                    "동시 사용자 처리",
                    f"{test['total_requests']}명 평균 지연시간",
                    round(test["avg_latency"], 2),
                    "seconds",
                    f"{test['total_requests']}명 동시"
                ])

    # Write CSV
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["구성", "테스트 유형", "측정 항목", "값", "단위", "세부사항"])
        writer.writerows(rows)

    print(f"✅ 결과 저장됨: {filename}")

def main():
    print("="*70)
    print("🚀 SGLang Qwen-Style Performance Benchmark")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Test configurations
    configs = [
        {"port": 8000, "name": "Baseline-Triton"},
        {"port": 8003, "name": "Balanced-v2-LOF"}
    ]

    all_results = []

    for config in configs:
        # Check if accessible
        try:
            response = requests.get(f"http://localhost:{config['port']}/health", timeout=2)
            if response.status_code == 200:
                tester = SGLangPerformanceTester(f"http://localhost:{config['port']}", config['name'])
                result = tester.run_comprehensive_test()
                all_results.append(result)
                print(f"  ✅ {config['name']} 테스트 완료")
        except Exception as e:
            print(f"  ❌ {config['name']} 접속 불가: {e}")

    # Save results
    if all_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sglang_qwen_style_benchmark_{timestamp}.csv'
        save_results_to_csv(all_results, filename)

        # Print comparison summary
        print("\n" + "="*70)
        print("📊 성능 비교 요약")
        print("="*70)

        for result in all_results:
            config = result["configuration"]

            # Calculate averages
            if "token_generation" in result:
                avg_tps = statistics.mean([t["tokens_per_second"] for t in result["token_generation"]])
                print(f"\n{config}:")
                print(f"  평균 토큰 생성 속도: {avg_tps:.2f} tokens/sec")

            if "throughput" in result:
                for t in result["throughput"]:
                    if t["total_requests"] == 10:
                        print(f"  10명 동시 처리량: {t['requests_per_second']:.2f} req/sec")
                        print(f"  10명 평균 지연시간: {t['avg_latency']:.2f} seconds")

if __name__ == '__main__':
    main()