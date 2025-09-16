#!/bin/bash
# Alternative: vLLM with latest PyTorch for RTX 5090

set -e

echo "=== vLLM Alternative Deployment ==="
echo "If SGLang fails, use latest vLLM with PyTorch 2.7.0"
echo "===================================="

# Clean up
docker stop vllm-rtx5090 2>/dev/null || true
docker rm vllm-rtx5090 2>/dev/null || true

# Run vLLM with PyTorch 2.7.0 support
docker run -d \
  --name vllm-rtx5090 \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e VLLM_ATTENTION_BACKEND=XFORMERS \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  --shm-size 32g \
  --ulimit memlock=-1 \
  vllm/vllm-openai:v0.6.4.post1 \
  --model Qwen/Qwen3-32B-AWQ \
  --quantization awq_marlin \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --enforce-eager

echo "✅ vLLM deployment started"
echo "Check logs: docker logs -f vllm-rtx5090"