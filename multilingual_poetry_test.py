#!/usr/bin/env python3
"""
Multilingual Poetry Generation Test
Tests token generation speed with a large multilingual output request
"""

import time
import requests
import json
from datetime import datetime

def test_multilingual_poetry(port=8000, max_tokens=5000):
    """Test generating multilingual poetry with high token count"""

    url = f"http://localhost:{port}/v1/completions"

    # Prompt for multilingual poetry
    prompt = """Write a long poem alternating between English, Korean, Japanese, and Chinese.
Each stanza should be in a different language, cycling through all four languages.
Start with English, then Korean, then Japanese, then Chinese, and repeat.
Make it a beautiful poem about the four seasons. Begin:

[English]
Spring arrives with gentle breeze,
Cherry blossoms dance on trees,

[한국어]
여름이 오면 뜨거운 햇살,
푸른 바다가 우리를 부르네,

[日本語]
秋の風が涼しくなり、
紅葉が山を彩る時、

[中文]
冬天带来白雪皑皑，
万物在寂静中等待，

Continue this pattern for many more stanzas:
"""

    print("🌏 Multilingual Poetry Generation Test")
    print("=" * 60)
    print(f"📝 Requesting {max_tokens} tokens of multilingual poetry")
    print(f"🔤 Languages: English, Korean, Japanese, Chinese")
    print("=" * 60)
    print()

    # Make request
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending request...")
    start_time = time.perf_counter()

    try:
        response = requests.post(
            url,
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.8,
                "top_p": 0.95,
                "frequency_penalty": 0.3,  # Reduce repetition
                "presence_penalty": 0.3,   # Encourage variety
            },
            timeout=300  # 5 minute timeout for large generation
        )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        if response.status_code == 200:
            data = response.json()

            # Get metrics
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Calculate performance
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0

            # Get generated text
            generated_text = ""
            if "choices" in data and data["choices"]:
                generated_text = data["choices"][0].get("text", "")

            # Count lines
            lines = generated_text.count('\n') + 1

            # Display results
            print("✅ Generation Complete!")
            print("=" * 60)
            print("📊 Performance Metrics:")
            print(f"  ⏱️  Total Time: {total_time:.2f} seconds")
            print(f"  📝 Prompt Tokens: {prompt_tokens}")
            print(f"  ✍️  Generated Tokens: {completion_tokens}")
            print(f"  📚 Total Tokens: {total_tokens}")
            print(f"  🚀 Speed: {tokens_per_second:.2f} tokens/second")
            print(f"  📏 Lines Generated: ~{lines}")
            print()

            # Show language distribution (approximate)
            english_count = generated_text.lower().count('the') + generated_text.lower().count('and')
            korean_count = generated_text.count('이') + generated_text.count('는') + generated_text.count('가')
            japanese_count = generated_text.count('の') + generated_text.count('は') + generated_text.count('が')
            chinese_count = generated_text.count('的') + generated_text.count('在') + generated_text.count('了')

            print("🌍 Language Distribution (approximate):")
            print(f"  🇬🇧 English indicators: {english_count}")
            print(f"  🇰🇷 Korean indicators: {korean_count}")
            print(f"  🇯🇵 Japanese indicators: {japanese_count}")
            print(f"  🇨🇳 Chinese indicators: {chinese_count}")
            print()

            # Show sample of generated text
            print("📜 Sample of Generated Poetry (first 1000 chars):")
            print("-" * 40)
            print(generated_text[:1000])
            print("-" * 40)
            print()

            # Show end sample
            if len(generated_text) > 2000:
                print("📜 Sample from End (last 500 chars):")
                print("-" * 40)
                print(generated_text[-500:])
                print("-" * 40)

            # Performance summary
            print()
            print("=" * 60)
            print("🎯 Performance Summary:")
            print(f"  {'✅ EXCELLENT' if tokens_per_second > 50 else '⚠️ MODERATE' if tokens_per_second > 20 else '❌ SLOW'}")
            print(f"  Token Generation Speed: {tokens_per_second:.2f} tok/s")
            print(f"  Time per 1000 tokens: {(1000/tokens_per_second):.2f} seconds")
            print(f"  Throughput: {completion_tokens/total_time*60:.0f} tokens/minute")

            # Save full output to file
            output_file = f"multilingual_poetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Multilingual Poetry Generation ===\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Tokens: {completion_tokens}\n")
                f.write(f"Time: {total_time:.2f}s\n")
                f.write(f"Speed: {tokens_per_second:.2f} tok/s\n")
                f.write("=" * 40 + "\n\n")
                f.write(generated_text)

            print(f"\n💾 Full output saved to: {output_file}")

        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text[:500]}")

    except requests.exceptions.Timeout:
        print("❌ Request timed out after 5 minutes")
        print("The model may be struggling with such a large generation request")

    except Exception as e:
        print(f"❌ Error occurred: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")

def main():
    import sys

    # Check if custom token count specified
    max_tokens = 5000
    if len(sys.argv) > 1:
        try:
            max_tokens = int(sys.argv[1])
        except:
            print(f"Using default max_tokens: {max_tokens}")

    print("🚀 Starting Multilingual Poetry Generation Test")
    print(f"📊 Target: {max_tokens} tokens")
    print()

    # Check server
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print("✅ Server is running\n")
    except:
        print("⚠️  Server health check failed, but continuing...\n")

    # Run test
    test_multilingual_poetry(max_tokens=max_tokens)

if __name__ == "__main__":
    main()