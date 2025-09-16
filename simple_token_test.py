#!/usr/bin/env python3

import time
import requests
import json

# Simple token speed test
def test_token_speed(port=8000):
    url = f"http://localhost:{port}/v1/completions"

    test_cases = [
        ("What is 2+2?", 10),
        ("Tell me a short story", 50),
        ("Explain quantum computing", 100),
    ]

    print("🚀 Token Speed Test for SGLang")
    print("=" * 50)

    for prompt, max_tokens in test_cases:
        print(f"\n📝 Testing: '{prompt[:30]}...' (max_tokens={max_tokens})")

        start_time = time.perf_counter()

        try:
            response = requests.post(
                url,
                json={
                    "model": "Qwen/Qwen3-8B",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=60
            )

            end_time = time.perf_counter()

            if response.status_code == 200:
                data = response.json()
                total_time = end_time - start_time

                # Get token counts
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", max_tokens)

                # Calculate tokens per second
                tokens_per_second = completion_tokens / total_time if total_time > 0 else 0

                print(f"✅ Success!")
                print(f"   Tokens generated: {completion_tokens}")
                print(f"   Time taken: {total_time:.2f} seconds")
                print(f"   Speed: {tokens_per_second:.2f} tokens/second")

                # Show sample output
                if "choices" in data and data["choices"]:
                    output = data["choices"][0].get("text", "")
                    print(f"   Output: {output[:100]}...")
            else:
                print(f"❌ Failed: {response.status_code}")
                print(f"   Error: {response.text[:200]}")

        except requests.exceptions.Timeout:
            print("❌ Request timed out")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server")
        except Exception as e:
            print(f"❌ Error: {e}")

        time.sleep(2)  # Pause between tests

    print("\n" + "=" * 50)
    print("✅ Test completed!")

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

    # Check if server is running
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        print(f"✅ Server is running on port {port}\n")
    except:
        print(f"⚠️  Server may not be ready on port {port}, trying anyway...\n")

    test_token_speed(port)