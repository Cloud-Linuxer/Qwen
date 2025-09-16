#!/bin/bash
# Qwen3-32B-AWQ 모델 배포 스크립트 (SGLang)
# AWQ 4비트 양자화 모델 - RTX 5090 최적화

set -e

echo "=== Qwen3-32B-AWQ 배포 스크립트 ==="
echo "4비트 양자화 모델 - 메모리 효율적"
echo "RTX 5090 32GB VRAM 최적화"
echo "======================================"
echo

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 모델 정보
MODEL="Qwen/Qwen3-32B-AWQ"
MODEL_NAME="qwen3-32b-awq"
QUANTIZATION="awq"

echo -e "${BLUE}📦 모델 정보:${NC}"
echo "• 모델: Qwen3-32B-AWQ"
echo "• 크기: 약 16GB (4비트 양자화)"
echo "• 파라미터: 32B"
echo "• 양자화: AWQ (Activation-aware Weight Quantization)"
echo

# GPU 확인
echo "🔍 GPU 확인 중..."
if ! nvidia-smi &>/dev/null; then
    echo -e "${RED}❌ NVIDIA GPU를 찾을 수 없습니다${NC}"
    exit 1
fi

# GPU 정보 표시
echo -e "${BLUE}📊 GPU 정보:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
echo

# VRAM 확인
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo -e "${BLUE}💾 사용 가능한 VRAM: ${VRAM_FREE}MB${NC}"

# AWQ 모델은 약 16GB 필요
REQUIRED_VRAM=16000
if [ "$VRAM_FREE" -lt "$REQUIRED_VRAM" ]; then
    echo -e "${RED}❌ VRAM 부족. 필요: ${REQUIRED_VRAM}MB, 사용가능: ${VRAM_FREE}MB${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 충분한 VRAM 사용 가능${NC}"
echo

# 기존 컨테이너 확인 및 정리
echo "🔍 기존 컨테이너 확인 중..."
EXISTING=$(docker ps -a --format "{{.Names}}" | grep -E "sglang|qwen|vllm" || true)
if [ ! -z "$EXISTING" ]; then
    echo -e "${YELLOW}⚠️  기존 컨테이너 발견:${NC}"
    echo "$EXISTING"
    echo "정리 중..."
    echo "$EXISTING" | xargs -r docker stop 2>/dev/null || true
    echo "$EXISTING" | xargs -r docker rm 2>/dev/null || true
    echo -e "${GREEN}✅ 정리 완료${NC}"
fi

# Docker Compose 파일 생성
echo "📝 Docker Compose 설정 생성 중..."
cat > docker-compose.qwen3-32b-awq.yml << 'EOF'
version: '3.8'

services:
  qwen3-32b-awq:
    image: lmsysorg/sglang:latest
    container_name: qwen3-32b-awq-sglang
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "127.0.0.1:8000:8000"
      - "127.0.0.1:8001:8001"  # 메트릭스
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./models:/models
      - /tmp:/tmp
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HF_TOKEN=${HF_TOKEN:-}
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}
      - HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
      # RTX 5090 최적화
      - TORCH_CUDA_ARCH_LIST=7.0;7.5;8.0;8.6;8.9;9.0;12.0+PTX
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      # AWQ 양자화 설정
      - VLLM_USE_MODELSCOPE=0
      - QUANTIZATION_METHOD=awq
    command: >
      python -m sglang.launch_server
      --model-path Qwen/Qwen3-32B-AWQ
      --host 0.0.0.0
      --port 8000
      --dtype half
      --quantization awq
      --max-total-tokens 16384
      --mem-fraction-static 0.90
      --trust-remote-code
      --enable-torch-compile
      --cuda-graph-max-bs 32
      --chunked-prefill-size 512
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 300s
    restart: unless-stopped
    shm_size: '32gb'
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 65536
        hard: 65536

networks:
  default:
    name: qwen3-awq-network
    driver: bridge
EOF

echo -e "${GREEN}✅ Docker Compose 설정 생성 완료${NC}"
echo

# .env 파일 확인
if [ ! -f .env ]; then
    echo "📝 .env 파일 생성 중..."
    cat > .env << 'EOF'
# Hugging Face Token (선택사항)
HF_TOKEN=

# 오프라인 모드 (1로 설정시 캐시된 모델만 사용)
HF_HUB_OFFLINE=0
EOF
    echo -e "${YELLOW}ℹ️  .env 파일 생성됨. 필요시 HF_TOKEN을 추가하세요${NC}"
fi

# 서비스 시작
echo -e "${BLUE}🚀 Qwen3-32B-AWQ 서버 시작 중...${NC}"
docker compose -f docker-compose.qwen3-32b-awq.yml up -d

# 서비스 준비 대기
echo "⏳ 모델 로딩 중..."
echo "AWQ 모델은 처음 다운로드시 시간이 걸릴 수 있습니다 (약 16GB)..."
echo

# 로그 모니터링
echo "📋 실시간 로그 (Ctrl+C로 중단):"
echo "----------------------------------------"

# 백그라운드에서 헬스체크
(
    MAX_RETRIES=60
    RETRY_COUNT=0
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo
            echo -e "${GREEN}✅ 서버가 준비되었습니다!${NC}"
            break
        fi
        sleep 5
        RETRY_COUNT=$((RETRY_COUNT + 1))
    done
) &

# 로그 표시
timeout 300 docker compose -f docker-compose.qwen3-32b-awq.yml logs -f 2>&1 | while IFS= read -r line; do
    if echo "$line" | grep -q "error\|ERROR\|failed\|FAILED"; then
        echo -e "${RED}$line${NC}"
    elif echo "$line" | grep -q "warning\|WARNING"; then
        echo -e "${YELLOW}$line${NC}"
    elif echo "$line" | grep -q "Model loaded\|Server started\|Ready"; then
        echo -e "${GREEN}$line${NC}"
        break
    else
        echo "$line"
    fi
done || true

echo
echo "=========================================="
echo

# 서비스 상태 확인
docker compose -f docker-compose.qwen3-32b-awq.yml ps

echo
echo -e "${BLUE}=== API 엔드포인트 ===${NC}"
echo "📍 메인 API: http://localhost:8000"
echo "📍 헬스체크: http://localhost:8000/health"
echo "📍 모델 정보: http://localhost:8000/v1/models"
echo "📍 메트릭스: http://localhost:8001/metrics"
echo

# API 테스트
echo -e "${BLUE}🧪 API 테스트 중...${NC}"
sleep 5

# Completion 테스트
TEST_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-32B-AWQ",
        "prompt": "AWQ 양자화의 장점은",
        "max_tokens": 100,
        "temperature": 0.7
    }' 2>/dev/null || echo "{}")

if echo "$TEST_RESPONSE" | jq -e '.choices[0].text' > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API 테스트 성공!${NC}"
    echo "응답: $(echo "$TEST_RESPONSE" | jq -r '.choices[0].text' | head -2)..."
else
    echo -e "${YELLOW}⚠️  API가 아직 준비 중입니다. 잠시 후 다시 시도하세요.${NC}"
fi

echo
echo -e "${BLUE}=== 유용한 명령어 ===${NC}"
echo "📊 GPU 모니터링:"
echo "  watch -n 1 nvidia-smi"
echo
echo "📝 로그 확인:"
echo "  docker compose -f docker-compose.qwen3-32b-awq.yml logs -f"
echo
echo "🔄 서비스 재시작:"
echo "  docker compose -f docker-compose.qwen3-32b-awq.yml restart"
echo
echo "🛑 서비스 중지:"
echo "  docker compose -f docker-compose.qwen3-32b-awq.yml down"
echo

echo -e "${BLUE}=== AWQ 모델 특징 ===${NC}"
echo "• 4비트 양자화로 메모리 사용량 50% 절감"
echo "• 32B 파라미터 모델을 16GB VRAM에서 실행"
echo "• 속도는 FP16 대비 약 90% 유지"
echo "• RTX 5090에서 최적 성능 발휘"
echo

echo -e "${GREEN}🎉 Qwen3-32B-AWQ 배포 완료!${NC}"