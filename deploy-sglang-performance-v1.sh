#!/bin/bash

# SGLang 성능 개선 v1: Triton Attention + 환경변수 최적화

echo "🚀 SGLang 성능 개선 v1 배포..."
echo "📋 최적화: Triton attention, CUDA 비동기, 메모리 증가"

# 기존 컨테이너 정리
docker stop sglang-perf-v1 2>/dev/null
docker rm sglang-perf-v1 2>/dev/null

# 최적화된 설정으로 배포
docker run -d \
  --name sglang-perf-v1 \
  --runtime nvidia \
  --gpus all \
  -p 8002:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  -e CUDA_LAUNCH_BLOCKING=0 \
  -e OMP_NUM_THREADS=8 \
  -e CUDA_DEVICE_ORDER="PCI_BUS_ID" \
  --shm-size 32g \
  sglang:blackwell-final-v2 \
  --model-path Qwen/Qwen3-32B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --max-total-tokens 3072 \
  --max-prefill-tokens 1536 \
  --chunked-prefill-size 1024 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --decode-log-interval 40

echo "✅ 컨테이너 시작. 로그 확인..."
sleep 5
docker logs sglang-perf-v1 --tail 20