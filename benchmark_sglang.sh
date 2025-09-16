#!/bin/bash

echo "=== SGLang 성능 벤치마크 ==="
echo ""

# 1. 짧은 응답 테스트 (5회 평균)
echo "📊 Test 1: 짧은 응답 (10 토큰)"
total_time=0
for i in {1..5}; do
  start=$(date +%s%N)
  curl -s -X POST http://localhost:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3-32B-AWQ",
      "prompt": "The capital of France is",
      "max_tokens": 10,
      "temperature": 0.1
    }' > /dev/null
  end=$(date +%s%N)
  elapsed=$((($end - $start) / 1000000))
  total_time=$(($total_time + $elapsed))
  echo "  시도 $i: ${elapsed}ms"
done
avg=$((total_time / 5))
echo "  ⏱️ 평균: ${avg}ms"
echo ""

# 2. 중간 응답 테스트
echo "📊 Test 2: 중간 응답 (50 토큰)"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Write a paragraph about artificial intelligence:",
    "max_tokens": 50,
    "temperature": 0.3
  }')
end=$(date +%s%N)
elapsed=$((($end - $start) / 1000000))
tokens=$(echo "$response" | jq -r '.usage.completion_tokens')
tps=$(echo "scale=2; $tokens * 1000 / $elapsed" | bc)
echo "  시간: ${elapsed}ms"
echo "  토큰: $tokens"
echo "  ⚡ 속도: ${tps} tokens/s"
echo ""

# 3. 긴 응답 테스트
echo "📊 Test 3: 긴 응답 (100 토큰)"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Explain the theory of relativity in simple terms:",
    "max_tokens": 100,
    "temperature": 0.3
  }')
end=$(date +%s%N)
elapsed=$((($end - $start) / 1000000))
tokens=$(echo "$response" | jq -r '.usage.completion_tokens')
tps=$(echo "scale=2; $tokens * 1000 / $elapsed" | bc)
echo "  시간: ${elapsed}ms"
echo "  토큰: $tokens"
echo "  ⚡ 속도: ${tps} tokens/s"
echo ""

# 4. 한국어 테스트
echo "📊 Test 4: 한국어 응답 (30 토큰)"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "messages": [{"role": "user", "content": "인공지능의 장점을 설명해주세요"}],
    "max_tokens": 30,
    "temperature": 0.3
  }')
end=$(date +%s%N)
elapsed=$((($end - $start) / 1000000))
tokens=$(echo "$response" | jq -r '.usage.completion_tokens')
tps=$(echo "scale=2; $tokens * 1000 / $elapsed" | bc)
echo "  시간: ${elapsed}ms"
echo "  토큰: $tokens"
echo "  ⚡ 속도: ${tps} tokens/s"
echo ""

echo "=== 벤치마크 완료 ==="