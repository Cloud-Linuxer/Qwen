#!/bin/bash
# RTX 5090 최종 배포 스크립트 - CUDA 12.8 + PyTorch 2.7.0
# Qwen3-32B-AWQ 모델 서빙

set -e

echo "=== RTX 5090 Blackwell 최종 배포 ==="
echo "CUDA 12.8 + PyTorch 2.7.0 정식 지원"
echo "====================================="
echo

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# GPU 확인
echo -e "${BLUE}📊 GPU 정보:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader
echo

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리..."
docker ps -aq --filter "name=rtx5090\|sglang\|qwen" | xargs -r docker stop 2>/dev/null || true
docker ps -aq --filter "name=rtx5090\|sglang\|qwen" | xargs -r docker rm 2>/dev/null || true
echo

# Docker 이미지 빌드
echo -e "${BLUE}🔨 RTX 5090 정식 지원 이미지 빌드 중...${NC}"
echo "CUDA 12.8 + PyTorch 2.7.0 사용"
docker build -f Dockerfile.rtx5090-cuda128 -t rtx5090-sglang:latest . || {
    echo -e "${RED}❌ 이미지 빌드 실패${NC}"
    exit 1
}
echo -e "${GREEN}✅ 이미지 빌드 완료!${NC}"
echo

# Docker Compose 파일 생성
echo "📝 Docker Compose 설정 생성..."
cat > docker-compose.rtx5090-final.yml << 'EOF'
services:
  rtx5090-sglang:
    image: rtx5090-sglang:latest
    container_name: rtx5090-qwen-awq
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
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    command:
      - --model-path=Qwen/Qwen3-32B-AWQ
      - --host=0.0.0.0
      - --port=8000
      - --dtype=half
      - --quantization=awq_marlin
      - --max-total-tokens=16384
      - --mem-fraction-static=0.95
      - --trust-remote-code
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
    name: rtx5090-network
    driver: bridge
EOF
echo -e "${GREEN}✅ 설정 완료${NC}"
echo

# .env 파일 생성
if [ ! -f .env ]; then
    cat > .env << 'EOF'
# Hugging Face Token (선택사항)
HF_TOKEN=

# 오프라인 모드 (1로 설정시 캐시된 모델만 사용)
HF_HUB_OFFLINE=0
EOF
    echo "📝 .env 파일 생성됨"
fi

# 서비스 시작
echo -e "${BLUE}🚀 SGLang 서버 시작...${NC}"
docker compose -f docker-compose.rtx5090-final.yml up -d

# 상태 확인
echo "⏳ 서버 초기화 대기중..."
sleep 10

# 로그 확인 (백그라운드)
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

# 실시간 로그 표시
echo
echo "📋 서버 로그:"
echo "----------------------------------------"
timeout 30 docker compose -f docker-compose.rtx5090-final.yml logs -f 2>&1 | while IFS= read -r line; do
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
done || true

echo
echo "=========================================="
echo

# 서비스 상태
docker compose -f docker-compose.rtx5090-final.yml ps

echo
echo -e "${BLUE}=== API 엔드포인트 ===${NC}"
echo "📍 메인 API: http://localhost:8000"
echo "📍 헬스체크: http://localhost:8000/health"
echo "📍 모델 정보: http://localhost:8000/v1/models"
echo

# API 테스트
echo -e "${BLUE}🧪 API 테스트...${NC}"
sleep 5
curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-32B-AWQ",
        "prompt": "RTX 5090의 장점은",
        "max_tokens": 50,
        "temperature": 0.7
    }' | jq -r '.choices[0].text' 2>/dev/null || echo "API 준비 중..."

echo
echo -e "${BLUE}=== 명령어 ===${NC}"
echo "로그: docker compose -f docker-compose.rtx5090-final.yml logs -f"
echo "중지: docker compose -f docker-compose.rtx5090-final.yml down"
echo "GPU: watch -n 1 nvidia-smi"
echo

echo -e "${GREEN}🎉 RTX 5090 배포 완료!${NC}"
echo "CUDA 12.8 + PyTorch 2.7.0 정식 지원"