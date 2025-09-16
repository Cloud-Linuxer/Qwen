#!/bin/bash
# Stable SGLang Deployment for RTX 5090

set -e

echo "=== Stable SGLang Deployment ==="
echo "Using PyTorch 2.7.0 + xformers (no FlashInfer)"
echo "=================================="

# Clean up
docker stop sglang-stable 2>/dev/null || true
docker rm sglang-stable 2>/dev/null || true

# Build stable image
echo "🔨 Building stable image..."
docker build -f Dockerfile.sglang-stable -t sglang:stable . || {
    echo "❌ Build failed"
    exit 1
}

echo "🚀 Starting SGLang with conservative settings..."
docker run -d \
  --name sglang-stable \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  -e TORCH_USE_CUDA_DSA=1 \
  -e CUDA_LAUNCH_BLOCKING=1 \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  sglang:stable \
  --model-path Qwen/Qwen3-32B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq_marlin \
  --max-total-tokens 4096 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend xformers \
  --log-level debug \
  --show-time-cost

echo "⏳ Waiting for server startup..."
sleep 10

# Monitor with detailed logging
echo "📋 Server logs:"
docker logs sglang-stable 2>&1 | tail -50

echo ""
echo "=== Debugging Commands ==="
echo "Logs: docker logs -f sglang-stable"
echo "Shell: docker exec -it sglang-stable bash"
echo "Debug: docker exec -it sglang-stable gdb python"
echo ""

# Health check
sleep 20
if curl -f http://localhost:8000/health 2>/dev/null; then
    echo "✅ Server is healthy!"
else
    echo "⚠️ Server may still be starting..."
    echo "Check logs: docker logs -f sglang-stable"
fi