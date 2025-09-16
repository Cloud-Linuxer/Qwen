#!/bin/bash
# SGLang with PyTorch 2.7.0 for RTX 5090

set -e

echo "=== RTX 5090용 SGLang 배포 ==="
echo "PyTorch 2.7.0 업그레이드 방식"
echo "================================"

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# GPU 확인
echo -e "${BLUE}📊 GPU 정보:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리..."
docker stop sglang-rtx5090 2>/dev/null || true
docker rm sglang-rtx5090 2>/dev/null || true
echo

# SGLang 직접 실행 (PyTorch 업그레이드 포함)
echo -e "${BLUE}🚀 SGLang 컨테이너 시작...${NC}"
docker run -d \
  --name sglang-rtx5090 \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;12.0+PTX" \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint /bin/bash \
  lmsysorg/sglang:latest \
  -c "pip install --upgrade torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
      python -m sglang.launch_server \
      --model-path Qwen/Qwen3-32B-AWQ \
      --host 0.0.0.0 \
      --port 8000 \
      --dtype half \
      --quantization awq_marlin \
      --max-total-tokens 16384 \
      --mem-fraction-static 0.95 \
      --trust-remote-code \
      --chunked-prefill-size 1024 \
      --enable-torch-compile"

echo "⏳ 서버 초기화 대기중..."
sleep 10

# 로그 확인
echo -e "${BLUE}📋 서버 로그:${NC}"
docker logs sglang-rtx5090 --tail 50

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

echo -e "${GREEN}🎉 배포 시작됨!${NC}"