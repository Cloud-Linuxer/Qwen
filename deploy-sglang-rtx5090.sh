#!/bin/bash
# SGLang RTX 5090 호환 배포 스크립트
# Qwen3-32B-AWQ 모델 서빙

set -e

echo "=== SGLang RTX 5090 배포 ==="
echo "PyTorch 2.5.1 + CUDA 12.4 사용"
echo "================================"
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

echo -e "${BLUE}📦 모델 정보:${NC}"
echo "• 모델: Qwen3-32B-AWQ"
echo "• 양자화: AWQ 4-bit"
echo "• 메모리: ~16GB VRAM"
echo

# GPU 확인
echo "🔍 GPU 확인 중..."
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader
echo

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker ps -aq --filter "name=sglang\|qwen" | xargs -r docker stop 2>/dev/null || true
docker ps -aq --filter "name=sglang\|qwen" | xargs -r docker rm 2>/dev/null || true
echo

# Docker 이미지 빌드
echo -e "${BLUE}🔨 RTX 5090 호환 이미지 빌드 중...${NC}"
echo "첫 빌드는 10-15분 소요될 수 있습니다."
docker build -f Dockerfile.sglang-rtx5090 -t sglang-rtx5090:latest . || {
    echo -e "${RED}❌ 이미지 빌드 실패${NC}"
    exit 1
}
echo -e "${GREEN}✅ 이미지 빌드 완료${NC}"
echo

# Docker Compose 파일 생성
echo "📝 Docker Compose 설정 생성 중..."
cat > docker-compose.sglang-rtx5090.yml << 'EOF'
services:
  sglang-rtx5090:
    image: sglang-rtx5090:latest
    container_name: sglang-qwen3-awq
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
      - "127.0.0.1:8001:8001"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./models:/models
      - /tmp:/tmp
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HF_TOKEN=${HF_TOKEN:-}
      - TORCH_CUDA_ARCH_LIST=7.0;7.5;8.0;8.6;8.9;9.0;12.0+PTX
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    command:
      - --model-path=Qwen/Qwen3-32B-AWQ
      - --host=0.0.0.0
      - --port=8000
      - --dtype=half
      - --quantization=awq_marlin
      - --max-total-tokens=16384
      - --mem-fraction-static=0.95
      - --trust-remote-code
      - --cuda-graph-max-bs=16
      - --chunked-prefill-size=1024
      - --enable-torch-compile
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
    name: sglang-rtx5090-network
    driver: bridge
EOF
echo -e "${GREEN}✅ Docker Compose 설정 완료${NC}"
echo

# 서비스 시작
echo -e "${BLUE}🚀 SGLang 서버 시작 중...${NC}"
docker compose -f docker-compose.sglang-rtx5090.yml up -d

# 로그 모니터링
echo "📋 서버 시작 로그:"
echo "----------------------------------------"
timeout 60 docker compose -f docker-compose.sglang-rtx5090.yml logs -f 2>&1 | while IFS= read -r line; do
    if echo "$line" | grep -q "Model loaded\|Server started\|ready\|8000"; then
        echo -e "${GREEN}$line${NC}"
        break
    elif echo "$line" | grep -q "error\|ERROR\|failed"; then
        echo -e "${RED}$line${NC}"
    elif echo "$line" | grep -q "warning\|WARNING"; then
        echo -e "${YELLOW}$line${NC}"
    else
        echo "$line"
    fi
done &

# 헬스체크 대기
echo
echo "⏳ 서버 준비 대기 중..."
MAX_RETRIES=60
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ 서버가 준비되었습니다!${NC}"
        break
    fi
    echo -n "."
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ 서버 시작 실패${NC}"
    echo "로그 확인:"
    docker compose -f docker-compose.sglang-rtx5090.yml logs --tail=50
    exit 1
fi

echo
echo "=========================================="
echo

# API 테스트
echo -e "${BLUE}🧪 API 테스트${NC}"
curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-32B-AWQ",
        "prompt": "RTX 5090의 특징은",
        "max_tokens": 50,
        "temperature": 0.7
    }' | jq -r '.choices[0].text' 2>/dev/null || echo "API 준비 중..."

echo
echo -e "${BLUE}=== 서비스 정보 ===${NC}"
echo "📍 API: http://localhost:8000"
echo "📍 메트릭스: http://localhost:8001/metrics"
echo

echo -e "${BLUE}=== 명령어 ===${NC}"
echo "로그: docker compose -f docker-compose.sglang-rtx5090.yml logs -f"
echo "중지: docker compose -f docker-compose.sglang-rtx5090.yml down"
echo "GPU: watch -n 1 nvidia-smi"
echo

echo -e "${GREEN}🎉 배포 완료!${NC}"