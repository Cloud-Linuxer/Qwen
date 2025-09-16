#!/bin/bash

echo "═══════════════════════════════════════════════════════════════════"
echo "                 SGLang 최적화 성능 비교 테스트                     "
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "테스트 대상: 울트라 최적화 버전 (포트 8001)"
echo "환경: RTX 5090 + Qwen3-32B-AWQ + Torch Compile + LPM Schedule"
echo "시간: $(date)"
echo ""
echo "───────────────────────────────────────────────────────────────────"

# 1. 워밍업
echo "🔥 워밍업 중..."
curl -s -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-32B-AWQ", "prompt": "Hi", "max_tokens": 5}' > /dev/null
sleep 3

# 2. 짧은 응답 속도 테스트 (10회 평균)
echo ""
echo "📊 Test 1: 짧은 응답 속도 (10 토큰, 10회 평균)"
echo "───────────────────────────────────────────────────────────────────"
total=0
min=999999
max=0
for i in {1..10}; do
  start=$(date +%s%N)
  curl -s -X POST http://localhost:8001/v1/completions \
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
echo "  🔄 기존 대비: $(echo "scale=1; ($avg - 1049) * 100 / 1049" | bc)% 변화"

# 3. 처리량 테스트 (다양한 길이)
echo ""
echo "📊 Test 2: 처리량 측정 (다양한 길이)"
echo "───────────────────────────────────────────────────────────────────"

# 30 토큰
echo "  [30 토큰 생성]"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:8001/v1/completions \
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
echo "    🔄 기존 10.2 tok/s 대비: $(echo "scale=1; ($tps - 10.2) * 100 / 10.2" | bc)% 변화"

# 50 토큰
echo "  [50 토큰 생성]"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:8001/v1/completions \
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
tps50=$tps

# 100 토큰
echo "  [100 토큰 생성]"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:8001/v1/completions \
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
tps100=$tps

# 4. 스트리밍 테스트 (TTFT)
echo ""
echo "📊 Test 3: 스트리밍 성능 (첫 토큰까지 시간)"
echo "───────────────────────────────────────────────────────────────────"
start=$(date +%s%N)
curl -N -X POST http://localhost:8001/v1/completions \
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
echo "  🔄 기존 283ms 대비: $(echo "scale=1; ($ttft - 283) * 100 / 283" | bc)% 변화"

# 5. Torch Compile 효과 측정
echo ""
echo "📊 Test 4: Torch Compile 최적화 효과"
echo "───────────────────────────────────────────────────────────────────"
echo "  첫 실행 (컴파일):"
start=$(date +%s%N)
curl -s -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-32B-AWQ", "prompt": "Test compile:", "max_tokens": 20}' > /dev/null
end=$(date +%s%N)
first=$((($end - $start) / 1000000))
echo "    시간: ${first}ms"

echo "  두번째 실행 (컴파일 캐시):"
start=$(date +%s%N)
curl -s -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-32B-AWQ", "prompt": "Test compile:", "max_tokens": 20}' > /dev/null
end=$(date +%s%N)
second=$((($end - $start) / 1000000))
echo "    시간: ${second}ms"
echo "  ⚡ 컴파일 효과: $(echo "scale=1; ($first - $second) * 100 / $first" | bc)% 개선"

# 6. 연속 디코드 스텝 효과
echo ""
echo "📊 Test 5: 연속 디코드 스텝 (3 steps) 효과"
echo "───────────────────────────────────────────────────────────────────"
start=$(date +%s%N)
response=$(curl -s -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Generate a long text about space exploration:",
    "max_tokens": 150,
    "temperature": 0.3
  }')
end=$(date +%s%N)
elapsed=$((($end - $start) / 1000000))
tokens=$(echo "$response" | jq -r '.usage.completion_tokens')
tps=$(echo "scale=2; $tokens * 1000 / $elapsed" | bc)
echo "  150 토큰 생성: ${elapsed}ms | 속도: ${tps} tok/s"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "                      최적화 결과 요약                              "
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📊 처리량 개선:"
echo "  - 평균 속도: ~${tps50} tok/s (50 토큰 기준)"
echo "  - 기존 대비: $(echo "scale=1; ($tps50 - 10.2) * 100 / 10.2" | bc)% 변화"
echo ""
echo "⚡ 최적화 효과:"
echo "  - Torch Compile: ✅ 활성화"
echo "  - LPM Schedule: ✅ 적용"
echo "  - 연속 디코드 3 스텝: ✅ 적용"
echo "  - Triton FP32 감소: ✅ 적용"
echo ""

# GPU 상태 확인
echo "📊 GPU 상태:"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | while read line; do
  echo "  $line"
done

echo ""
echo "✅ 벤치마크 완료!"