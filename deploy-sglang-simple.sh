#!/bin/bash
# SGLang 간단 배포 - Qwen3-32B-AWQ

set -e

echo "=== SGLang AWQ 모델 배포 ==="
echo

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리..."
docker ps -aq --filter "name=sglang" | xargs -r docker stop 2>/dev/null || true
docker ps -aq --filter "name=sglang" | xargs -r docker rm 2>/dev/null || true

# Docker Compose 생성
cat > docker-compose.sglang-awq.yml << 'EOF'
services:
  sglang-awq:
    image: lmsysorg/sglang:latest
    container_name: sglang-qwen-awq
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
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./models:/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HF_TOKEN=${HF_TOKEN:-}
      # RTX 5090 workaround
      - TORCH_CUDA_ARCH_LIST=7.0;7.5;8.0;8.6;8.9;9.0+PTX
      - CUDA_LAUNCH_BLOCKING=0
    command:
      - --model-path=Qwen/Qwen3-32B-AWQ
      - --host=0.0.0.0
      - --port=8000
      - --dtype=half
      - --quantization=awq_marlin
      - --max-total-tokens=16384
      - --mem-fraction-static=0.95
      - --trust-remote-code
      - --disable-cuda-graph
      - --disable-custom-all-reduce
    restart: unless-stopped
    shm_size: '32gb'

networks:
  default:
    name: sglang-net
EOF

echo "🚀 서버 시작..."
docker compose -f docker-compose.sglang-awq.yml up -d

echo "⏳ 모델 로딩 대기..."
sleep 10

echo "📋 로그 확인:"
docker compose -f docker-compose.sglang-awq.yml logs --tail=50

echo
echo "✅ 배포 시도 완료"
echo "명령어:"
echo "  로그: docker compose -f docker-compose.sglang-awq.yml logs -f"
echo "  중지: docker compose -f docker-compose.sglang-awq.yml down"