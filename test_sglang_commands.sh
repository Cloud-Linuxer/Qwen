#!/bin/bash

# SGLang API 테스트 명령어 모음
# RTX 5090 + Qwen3-32B-AWQ 모델

echo "=== SGLang API 테스트 스크립트 ==="
echo ""

# 1. 헬스체크
echo "1️⃣ 헬스체크 테스트..."
curl -s http://localhost:8000/health | jq '.'
echo ""

# 2. 모델 정보 확인
echo "2️⃣ 모델 정보 확인..."
curl -s http://localhost:8000/get_model_info | jq '.'
echo ""

# 3. 간단한 완성 테스트
echo "3️⃣ 텍스트 완성 테스트 (영어)..."
curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "temperature": 0.1
  }' | jq '.choices[0].text'
echo ""

# 4. 한국어 채팅 테스트
echo "4️⃣ 한국어 채팅 테스트..."
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "messages": [
      {"role": "user", "content": "대한민국의 수도는 어디인가요?"}
    ],
    "max_tokens": 30,
    "temperature": 0.1
  }' | jq '.choices[0].message.content'
echo ""

# 5. 코드 생성 테스트
echo "5️⃣ 코드 생성 테스트..."
curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "def quicksort(arr):\n    # Python implementation of quicksort\n",
    "max_tokens": 100,
    "temperature": 0.2
  }' | jq '.choices[0].text'
echo ""

# 6. 스트리밍 테스트
echo "6️⃣ 스트리밍 테스트 (실시간 출력)..."
curl -N -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Write a haiku about AI:",
    "max_tokens": 50,
    "temperature": 0.7,
    "stream": true
  }'
echo ""
echo ""

# 7. 배치 처리 테스트
echo "7️⃣ 배치 처리 테스트 (여러 프롬프트)..."
curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": ["2+2=", "The sky is", "Python is a"],
    "max_tokens": 10,
    "temperature": 0.1
  }' | jq '.'
echo ""

# 8. 긴 컨텍스트 테스트
echo "8️⃣ 긴 컨텍스트 테스트..."
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant who answers questions about the given context."},
      {"role": "user", "content": "Context: The RTX 5090 is NVIDIAs latest flagship GPU based on the Blackwell architecture. It features 32GB of VRAM and uses the sm_120 compute capability. The card has impressive performance but requires special configuration for frameworks like SGLang due to its new architecture. Question: What architecture is the RTX 5090 based on?"}
    ],
    "max_tokens": 50,
    "temperature": 0.1
  }' | jq '.choices[0].message.content'
echo ""

# 9. 다양한 온도 설정 테스트
echo "9️⃣ 창의성 테스트 (temperature=0.9)..."
curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Once upon a time in a digital world,",
    "max_tokens": 50,
    "temperature": 0.9,
    "top_p": 0.95
  }' | jq '.choices[0].text'
echo ""

# 10. 성능 벤치마크
echo "🔟 간단한 성능 측정..."
START_TIME=$(date +%s%N)
curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Hello",
    "max_tokens": 100,
    "temperature": 0.5
  }' > /dev/null
END_TIME=$(date +%s%N)
ELAPSED=$((($END_TIME - $START_TIME) / 1000000))
echo "응답 시간: ${ELAPSED}ms"
echo ""

echo "=== 테스트 완료 ==="