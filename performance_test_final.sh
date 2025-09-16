#!/bin/bash

echo "═══════════════════════════════════════════════════════════════════"
echo "                    SGLang 최종 성능 테스트                         "
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "환경: RTX 5090 (32GB) + Qwen3-32B-AWQ + Triton Attention"
echo "시간: $(date)"
echo ""
echo "───────────────────────────────────────────────────────────────────"

# 1. 워밍업
echo "🔥 워밍업 중..."
curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-32B-AWQ", "prompt": "Hi", "max_tokens": 5}' > /dev/null
sleep 2

# 2. 짧은 응답 속도 테스트
echo ""
echo "📊 Test 1: 짧은 응답 속도 (10 토큰, 10회 평균)"
echo "───────────────────────────────────────────────────────────────────"
total=0
min=999999
max=0
for i in {1..10}; do
  start=$(date +%s%N)
  curl -s -X POST http://localhost:8000/v1/completions \
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

  if [ $elapsed -lt $min ]; then min=$elapsed; fi
  if [ $elapsed -gt $max ]; then max=$elapsed; fi

  printf "  시도 %2d: %4dms\n" $i $elapsed
done
avg=$((total / 10))
echo "───────────────────────────────────────────────────────────────────"
echo "  📈 평균: ${avg}ms | 최소: ${min}ms | 최대: ${max}ms"

# 3. 처리량 테스트 (tokens/s)
echo ""
echo "📊 Test 2: 처리량 측정 (다양한 길이)"
echo "───────────────────────────────────────────────────────────────────"

# 30 토큰
echo "  [30 토큰 생성]"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Explain quantum computing:",
    "max_tokens": 30,
    "temperature": 0.3
  }')
end=$(date +%s%N)
elapsed=$((($end - $start) / 1000000))
tokens=$(echo "$response" | jq -r '.usage.completion_tokens')
tps=$(echo "scale=2; $tokens * 1000 / $elapsed" | bc)
echo "    시간: ${elapsed}ms | 토큰: $tokens | 속도: ${tps} tok/s"

# 50 토큰
echo "  [50 토큰 생성]"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:8000/v1/completions \
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
echo "    시간: ${elapsed}ms | 토큰: $tokens | 속도: ${tps} tok/s"

# 100 토큰
echo "  [100 토큰 생성]"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Describe the future of technology:",
    "max_tokens": 100,
    "temperature": 0.3
  }')
end=$(date +%s%N)
elapsed=$((($end - $start) / 1000000))
tokens=$(echo "$response" | jq -r '.usage.completion_tokens')
tps=$(echo "scale=2; $tokens * 1000 / $elapsed" | bc)
echo "    시간: ${elapsed}ms | 토큰: $tokens | 속도: ${tps} tok/s"

# 4. 스트리밍 테스트
echo ""
echo "📊 Test 3: 스트리밍 성능 (첫 토큰까지 시간)"
echo "───────────────────────────────────────────────────────────────────"
start=$(date +%s%N)
curl -N -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Tell me a story:",
    "max_tokens": 20,
    "temperature": 0.5,
    "stream": true
  }' 2>/dev/null | head -n 1 > /dev/null
end=$(date +%s%N)
ttft=$((($end - $start) / 1000000))
echo "  ⏱️ Time to First Token (TTFT): ${ttft}ms"

# 5. 한국어 테스트
echo ""
echo "📊 Test 4: 한국어 처리 성능"
echo "───────────────────────────────────────────────────────────────────"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "messages": [{"role": "user", "content": "인공지능이 인류에게 미치는 영향을 설명해주세요"}],
    "max_tokens": 50,
    "temperature": 0.3
  }')
end=$(date +%s%N)
elapsed=$((($end - $start) / 1000000))
tokens=$(echo "$response" | jq -r '.usage.completion_tokens')
tps=$(echo "scale=2; $tokens * 1000 / $elapsed" | bc)
echo "  시간: ${elapsed}ms | 토큰: $tokens | 속도: ${tps} tok/s"

# 6. 긴 컨텍스트 테스트
echo ""
echo "📊 Test 5: 긴 입력 컨텍스트 처리"
echo "───────────────────────────────────────────────────────────────────"
long_prompt="Summarize the following text: The development of artificial intelligence has been one of the most significant technological advances in recent history. From its humble beginnings in the 1950s with simple rule-based systems, AI has evolved into sophisticated deep learning models that can understand natural language, recognize images, and even generate creative content. The implications for society are profound, affecting everything from healthcare and education to transportation and entertainment. As we stand on the brink of even more revolutionary advances, questions about ethics, safety, and the future of human work become increasingly important. What are the key points?"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen/Qwen3-32B-AWQ\",
    \"prompt\": \"$long_prompt\",
    \"max_tokens\": 30,
    \"temperature\": 0.3
  }")
end=$(date +%s%N)
elapsed=$((($end - $start) / 1000000))
prompt_tokens=$(echo "$response" | jq -r '.usage.prompt_tokens')
completion_tokens=$(echo "$response" | jq -r '.usage.completion_tokens')
echo "  입력 토큰: $prompt_tokens | 출력 토큰: $completion_tokens"
echo "  총 시간: ${elapsed}ms"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "                        테스트 완료                                "
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# GPU 상태 확인
echo "📊 GPU 상태:"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | while read line; do
  echo "  $line"
done

echo ""
echo "✅ 모든 테스트 완료!"