#!/bin/bash

# SGLang RTX 5090 Blackwell Final Deployment Script
# Build 완료된 이미지로 배포

echo "🚀 Starting SGLang RTX 5090 Blackwell Final Deployment..."

# 기존 컨테이너 정리
docker stop sglang-blackwell 2>/dev/null
docker rm sglang-blackwell 2>/dev/null

# 배포
docker run -d \
  --name sglang-blackwell \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e CUDA_LAUNCH_BLOCKING=1 \
  -e TORCH_SHOW_CPP_STACKTRACES=1 \
  --shm-size 16g \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  sglang:blackwell-final-v2 \
  --model-path Qwen/Qwen3-32B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --max-total-tokens 2048 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native

echo "✅ Container started. Checking logs..."
sleep 5
docker logs sglang-blackwell --tail 50