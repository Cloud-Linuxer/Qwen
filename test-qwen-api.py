#!/usr/bin/env python3

import requests
import json
import time

def test_sglang_api(port=8003):
    """Test SGLang API endpoint"""

    # API endpoint
    url = f"http://localhost:{port}/v1/chat/completions"

    # Test prompts
    test_cases = [
        {
            "name": "Simple Question",
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ]
        },
        {
            "name": "Code Generation",
            "messages": [
                {"role": "user", "content": "Write a Python function to calculate factorial"}
            ]
        },
        {
            "name": "Korean Test",
            "messages": [
                {"role": "user", "content": "한국의 수도는 어디인가요?"}
            ]
        }
    ]

    print(f"🔍 Testing SGLang API on port {port}...")
    print("=" * 50)

    for test_case in test_cases:
        print(f"\n📝 Test: {test_case['name']}")
        print(f"Question: {test_case['messages'][0]['content']}")

        payload = {
            "messages": test_case["messages"],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False
        }

        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=30)
            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                print(f"✅ Success (took {elapsed_time:.2f}s)")
                print(f"Answer: {answer[:200]}...")

                # Print token usage if available
                if 'usage' in result:
                    usage = result['usage']
                    print(f"Tokens: prompt={usage.get('prompt_tokens', 'N/A')}, "
                          f"completion={usage.get('completion_tokens', 'N/A')}, "
                          f"total={usage.get('total_tokens', 'N/A')}")
            else:
                print(f"❌ Failed with status {response.status_code}")
                print(f"Error: {response.text[:200]}")

        except requests.exceptions.ConnectionError:
            print("❌ Connection failed - server may not be running")
        except requests.exceptions.Timeout:
            print("❌ Request timed out after 30 seconds")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    print("\n" + "=" * 50)
    print("✨ Test completed!")

def check_server_health(port=8003):
    """Check if server is healthy"""
    try:
        # Try different possible endpoints
        endpoints = [
            f"http://localhost:{port}/health",
            f"http://localhost:{port}/v1/models",
            f"http://localhost:{port}/ping"
        ]

        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    print(f"✅ Server is healthy at {endpoint}")
                    return True
            except:
                continue

        print(f"⚠️  Server may be running but health endpoints not accessible")
        return False

    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return False

if __name__ == "__main__":
    port = 8003  # Using the running container's port

    print("🚀 SGLang API Test Suite")
    print("=" * 50)

    # Check server health first
    print("\n📋 Checking server health...")
    if check_server_health(port):
        print("Server appears to be running\n")
    else:
        print("Warning: Server health check failed, but will try API anyway\n")

    # Run API tests
    test_sglang_api(port)