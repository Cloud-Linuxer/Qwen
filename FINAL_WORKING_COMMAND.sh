#!/bin/bash
# FINAL WORKING SGLang deployment for RTX 5090
# This command has been tested and should work

docker stop sglang-final 2>/dev/null
docker rm sglang-final 2>/dev/null

echo "Starting SGLang for RTX 5090..."
echo "Using custom-built image with PyTorch 2.7.0"

docker run -d \
  --name sglang-rtx5090 \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e CUDA_LAUNCH_BLOCKING=1 \
  -e TORCHINDUCTOR_COMPILE_THREADS=1 \
  --shm-size 16g \
  --ulimit memlock=-1 \
  sglang:rtx5090-source \
  --model-path Qwen/Qwen3-32B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --max-total-tokens 1024 \
  --mem-fraction-static 0.65 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native \
  --log-level info

echo "Container started. Waiting for model to load..."
echo "This may take 2-3 minutes..."
echo ""
echo "Monitor with: docker logs -f sglang-rtx5090"
echo "Test with: curl http://localhost:8000/health"