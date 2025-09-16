#!/bin/bash

# SGLang RTX 5090 최적화 배포 스크립트
# AWQ_Marlin + 메모리 최적화 + 배치 처리 개선

echo "🚀 SGLang 최적화 배포 시작..."

# 기존 컨테이너 정리
docker stop sglang-optimized 2>/dev/null
docker rm sglang-optimized 2>/dev/null

# 최적화된 설정으로 배포
docker run -d \
  --name sglang-optimized \
  --runtime nvidia \
  --gpus all \
  -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e CUDA_LAUNCH_BLOCKING=0 \
  -e TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX" \
  --shm-size 32g \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  sglang:blackwell-final-v2 \
  --model-path Qwen/Qwen3-32B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq_marlin \
  --max-total-tokens 4096 \
  --max-prefill-tokens 2048 \
  --chunked-prefill-size 2048 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native \
  --decode-log-interval 10

echo "✅ 최적화 컨테이너 시작. 로그 확인 중..."
sleep 5
docker logs sglang-optimized --tail 50