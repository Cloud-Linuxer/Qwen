#!/bin/bash
# SGLang Deployment with Source-Built sgl_kernel for RTX 5090

set -e

echo "=== SGLang Deployment for RTX 5090 ==="
echo "Using source-built image with PyTorch 2.7.0"
echo "Model: Qwen3-32B-AWQ (4-bit quantized)"
echo "========================================"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# GPU check
echo -e "${BLUE}📊 GPU Information:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo

# Clean up existing containers
echo "🧹 Cleaning up existing containers..."
docker stop sglang-rtx5090 2>/dev/null || true
docker rm sglang-rtx5090 2>/dev/null || true
echo

# Run SGLang with our custom-built image
echo -e "${BLUE}🚀 Starting SGLang container...${NC}"
docker run -d \
  --name sglang-rtx5090 \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_LAUNCH_BLOCKING=0 \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc host \
  sglang:rtx5090-source \
  --model-path Qwen/Qwen3-32B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq_marlin \
  --max-total-tokens 8192 \
  --mem-fraction-static 0.90 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce

echo "⏳ 서버 초기화 대기중 (PyTorch 및 sgl-kernel 설치 중)..."
echo "이 과정은 5-10분 정도 소요될 수 있습니다."
sleep 30

# 로그 모니터링
echo -e "${BLUE}📋 서버 로그 모니터링 중...${NC}"
for i in {1..20}; do
  echo "확인 중... ($i/20)"
  if docker logs sglang-rtx5090 2>&1 | grep -q "Model loaded successfully"; then
    echo -e "${GREEN}✅ 모델 로드 완료!${NC}"
    break
  elif docker logs sglang-rtx5090 2>&1 | grep -q "Server started on"; then
    echo -e "${GREEN}✅ 서버 시작됨!${NC}"
    break
  elif docker logs sglang-rtx5090 2>&1 | tail -5 | grep -q "ERROR"; then
    echo -e "${RED}❌ 에러 발생:${NC}"
    docker logs sglang-rtx5090 2>&1 | tail -20
    break
  fi
  sleep 30
done

echo
echo "==========================================="
echo

# 상태 확인
docker ps -a | grep sglang-rtx5090

echo
echo -e "${BLUE}=== API 엔드포인트 ===${NC}"
echo "📍 메인 API: http://localhost:8000"
echo "📍 헬스체크: http://localhost:8000/health"
echo "📍 모델 정보: http://localhost:8000/v1/models"
echo

echo -e "${BLUE}=== 명령어 ===${NC}"
echo "로그: docker logs -f sglang-rtx5090"
echo "중지: docker stop sglang-rtx5090"
echo "재시작: docker restart sglang-rtx5090"
echo

# 최종 로그 확인
echo -e "${BLUE}=== 현재 상태 ===${NC}"
docker logs sglang-rtx5090 2>&1 | tail -20

echo
echo -e "${GREEN}🎉 배포 시작됨!${NC}"
echo "서버가 완전히 시작될 때까지 잠시 기다려주세요."