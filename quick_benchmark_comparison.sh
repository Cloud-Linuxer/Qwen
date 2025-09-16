#!/bin/bash

echo "════════════════════════════════════════════════════════════════"
echo "              SGLang 최적화 비교 - Quick Test                    "
echo "════════════════════════════════════════════════════════════════"
echo ""

# 포트 설정
PORT=$1
NAME=$2

if [ -z "$PORT" ]; then
  echo "Usage: $0 <port> <name>"
  echo "Example: $0 8003 'Balanced-v2'"
  exit 1
fi

echo "테스트 대상: $NAME (포트 $PORT)"
echo "────────────────────────────────────────────────────────────────"

# 워밍업
curl -s -X POST http://localhost:$PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-32B-AWQ", "prompt": "Hi", "max_tokens": 5}' > /dev/null
sleep 2

# 1. 짧은 응답 (5회 평균)
echo "📊 짧은 응답 테스트 (10 토큰, 5회):"
total=0
for i in {1..5}; do
  start=$(date +%s%N)
  curl -s -X POST http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3-32B-AWQ",
      "prompt": "The capital of France is",
      "max_tokens": 10,
      "temperature": 0.1
    }' > /dev/null
  end=$(date +%s%N)
  elapsed=$((($end - $start) / 1000000))
  total=$(($total + $elapsed))
  printf "  %dms" $elapsed
done
avg=$((total / 5))
echo ""
echo "  평균: ${avg}ms"

# 2. 50 토큰 처리량
echo ""
echo "📊 처리량 테스트 (50 토큰):"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:$PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Write about artificial intelligence:",
    "max_tokens": 50,
    "temperature": 0.3
  }')
end=$(date +%s%N)
elapsed=$((($end - $start) / 1000000))
tokens=$(echo "$response" | jq -r '.usage.completion_tokens')
tps=$(echo "scale=2; $tokens * 1000 / $elapsed" | bc)
echo "  시간: ${elapsed}ms | 토큰: $tokens | 속도: ${tps} tok/s"

# 3. TTFT
echo ""
echo "📊 TTFT (첫 토큰까지):"
start=$(date +%s%N)
curl -N -X POST http://localhost:$PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Tell me:",
    "max_tokens": 10,
    "stream": true
  }' 2>/dev/null | head -n 1 > /dev/null
end=$(date +%s%N)
ttft=$((($end - $start) / 1000000))
echo "  TTFT: ${ttft}ms"

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "결과 요약 [$NAME]:"
echo "  - 응답 지연: ${avg}ms"
echo "  - 처리 속도: ${tps} tok/s"
echo "  - TTFT: ${ttft}ms"
echo "════════════════════════════════════════════════════════════════"