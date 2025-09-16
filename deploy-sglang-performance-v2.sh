#!/bin/bash

# SGLang 성능 개선 v2: Triton + Torch Compile + CUDA Graphs (small batch)

echo "🚀 SGLang 성능 개선 v2 배포..."
echo "📋 최적화: Triton + Torch Compile + 작은 배치 CUDA Graphs"

# 기존 컨테이너 정리
docker stop sglang-perf-v2 2>/dev/null
docker rm sglang-perf-v2 2>/dev/null

# 최적화된 설정으로 배포
docker run -d \
  --name sglang-perf-v2 \
  --runtime nvidia \
  --gpus all \
  -p 8003:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  -e CUDA_LAUNCH_BLOCKING=0 \
  -e OMP_NUM_THREADS=8 \
  -e CUDA_DEVICE_ORDER="PCI_BUS_ID" \
  -e TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX" \
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
  --cuda-graph-max-bs 2 \
  --cuda-graph-bs 1 2 \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --enable-torch-compile \
  --torch-compile-max-bs 4 \
  --num-continuous-decode-steps 2 \
  --decode-log-interval 40

echo "✅ 컨테이너 시작. 로그 확인..."
sleep 5
docker logs sglang-perf-v2 --tail 20