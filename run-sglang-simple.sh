#!/bin/bash
# Simplest SGLang deployment for RTX 5090
# Uses pre-built image with all workarounds

set -e

echo "=== Simple SGLang Deployment for RTX 5090 ==="
echo "Using minimal configuration for stability"
echo "============================================="

# Clean up
docker stop sglang-simple 2>/dev/null || true
docker rm sglang-simple 2>/dev/null || true

# Use the already built image or fallback to official
if docker images | grep -q "sglang:rtx5090-source"; then
    IMAGE="sglang:rtx5090-source"
    echo "Using existing RTX 5090 image"
else
    IMAGE="lmsysorg/sglang:latest"
    echo "Using official SGLang image"
fi

# Run with absolute minimal configuration
docker run -d \
  --name sglang-simple \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e SGLANG_DISABLE_FLASHINFER=1 \
  -e SGLANG_DISABLE_CUDA_GRAPH=1 \
  -e DISABLE_CUSTOM_ALL_REDUCE=1 \
  -e CUDA_LAUNCH_BLOCKING=1 \
  --shm-size 16g \
  --ulimit memlock=-1 \
  $IMAGE \
  python -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --max-total-tokens 2048 \
  --mem-fraction-static 0.75 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --chunked-prefill-size 512 \
  --log-level info

echo "⏳ Waiting 30 seconds for startup..."
sleep 30

echo "📋 Server status:"
docker ps | grep sglang-simple || echo "Container not running"

echo ""
echo "📝 Logs (last 20 lines):"
docker logs sglang-simple 2>&1 | tail -20

echo ""
echo "Commands:"
echo "  Logs: docker logs -f sglang-simple"
echo "  Stop: docker stop sglang-simple"
echo ""

# Test
echo "🧪 Testing health endpoint..."
curl -s http://localhost:8000/health 2>/dev/null && echo "✅ Server is healthy!" || echo "⚠️ Server not ready yet"