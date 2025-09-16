#!/usr/bin/env python3
"""
Qwen Model Performance Test Script
Tests various aspects of model performance and saves results to CSV
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

class QwenPerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
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
            tokens_generated = len(result['choices'][0]['text'].split())
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
        print(f"Testing throughput with {num_requests} concurrent requests...")

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
        print("Testing token generation speed...")
        results = []

        prompt = "Write a detailed explanation about artificial intelligence"

        for max_tokens in token_counts:
            print(f"  Testing with max_tokens={max_tokens}...")
            result = self.test_single_request(prompt, max_tokens=max_tokens, temperature=0.7)
            if result["success"]:
                results.append({
                    "test_type": "token_generation",
                    "max_tokens": max_tokens,
                    "actual_tokens": result["tokens_generated"],
                    "latency": result["latency"],
                    "tokens_per_second": result["tokens_per_second"],
                    "response_length": result["response_length"]
                })

        return results

    def test_prompt_length_impact(self) -> List[Dict[str, Any]]:
        """Test impact of different prompt lengths on performance"""
        print("Testing prompt length impact...")
        results = []

        base_prompt = "Explain the concept of "
        topics = ["AI", "machine learning and deep learning",
                  "artificial intelligence, machine learning, deep learning, and neural networks",
                  "artificial intelligence, machine learning, deep learning, neural networks, natural language processing, computer vision, and reinforcement learning"]

        for topic in topics:
            prompt = base_prompt + topic
            prompt_tokens = len(prompt.split())

            result = self.test_single_request(prompt, max_tokens=100)
            if result["success"]:
                results.append({
                    "test_type": "prompt_length",
                    "prompt_tokens": prompt_tokens,
                    "latency": result["latency"],
                    "tokens_per_second": result["tokens_per_second"],
                    "total_tokens": prompt_tokens + result["tokens_generated"]
                })

        return results

    def test_temperature_variance(self) -> List[Dict[str, Any]]:
        """Test performance with different temperature settings"""
        print("Testing temperature variance...")
        results = []

        temperatures = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        prompt = "Generate a creative story about a robot"

        for temp in temperatures:
            print(f"  Testing with temperature={temp}...")
            result = self.test_single_request(prompt, max_tokens=100, temperature=temp)
            if result["success"]:
                results.append({
                    "test_type": "temperature",
                    "temperature": temp,
                    "latency": result["latency"],
                    "tokens_per_second": result["tokens_per_second"],
                    "response_length": result["response_length"]
                })

        return results

    def test_concurrent_users(self) -> List[Dict[str, Any]]:
        """Test with different numbers of concurrent users"""
        print("Testing concurrent users...")
        results = []

        user_counts = [1, 2, 5, 10, 20]

        for count in user_counts:
            print(f"  Testing with {count} concurrent users...")
            result = self.test_throughput(num_requests=count)
            results.append(result)

        return results

    def run_all_tests(self) -> None:
        """Run all performance tests"""
        print("Starting Qwen Model Performance Tests")
        print("=" * 50)

        all_results = []

        # Test 1: Token Generation Speed
        token_results = self.test_token_generation_speed()
        all_results.extend(token_results)

        # Test 2: Prompt Length Impact
        prompt_results = self.test_prompt_length_impact()
        all_results.extend(prompt_results)

        # Test 3: Temperature Variance
        temp_results = self.test_temperature_variance()
        all_results.extend(temp_results)

        # Test 4: Concurrent Users
        concurrent_results = self.test_concurrent_users()
        all_results.extend(concurrent_results)

        # Save results to CSV
        self.save_to_csv(all_results)

        # Print summary
        self.print_summary(all_results)

    def save_to_csv(self, results: List[Dict[str, Any]]) -> None:
        """Save test results to CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qwen_performance_test_{timestamp}.csv"

        # Collect all unique keys
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())

        # Write to CSV
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = sorted(list(all_keys))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"\nResults saved to: {filename}")

    def print_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print a summary of test results"""
        print("\n" + "=" * 50)
        print("Performance Test Summary")
        print("=" * 50)

        # Token generation summary
        token_tests = [r for r in results if r.get("test_type") == "token_generation"]
        if token_tests:
            avg_tps = statistics.mean([r["tokens_per_second"] for r in token_tests])
            print(f"\nToken Generation:")
            print(f"  Average tokens/second: {avg_tps:.2f}")
            print(f"  Best tokens/second: {max([r['tokens_per_second'] for r in token_tests]):.2f}")

        # Throughput summary
        throughput_tests = [r for r in results if r.get("test_type") == "throughput"]
        if throughput_tests:
            print(f"\nThroughput:")
            for test in throughput_tests:
                print(f"  {test['total_requests']} users: {test['requests_per_second']:.2f} req/s, "
                      f"avg latency: {test['avg_latency']:.2f}s")

        # Temperature impact
        temp_tests = [r for r in results if r.get("test_type") == "temperature"]
        if temp_tests:
            print(f"\nTemperature Impact:")
            print(f"  Temperature range tested: 0.0 - 1.0")
            avg_latency = statistics.mean([r["latency"] for r in temp_tests])
            print(f"  Average latency: {avg_latency:.2f}s")

def main():
    tester = QwenPerformanceTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()