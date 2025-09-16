#!/bin/bash

# SGLang Qwen3-8B Optimized Deployment
# Based on balanced-v2 configuration with Qwen3-8B model

echo "🚀 Deploying Qwen3-8B with SGLang Optimized Configuration"
echo "=================================================="
echo "📋 Configuration:"
echo "  ✅ Model: Qwen/Qwen3-8B (Hugging Face)"
echo "  ✅ Torch Compile Enabled"
echo "  ✅ LOF Scheduling Policy"
echo "  ✅ Triton Attention Backend"
echo "  ✅ 2-step Continuous Decode"
echo "=================================================="

# Stop and remove existing container
docker stop qwen3-8b-sglang 2>/dev/null
docker rm qwen3-8b-sglang 2>/dev/null

# Deploy Qwen3-8B with optimized settings
docker run -d \
  --name qwen3-8b-sglang \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256" \
  -e CUDA_LAUNCH_BLOCKING=0 \
  -e OMP_NUM_THREADS=10 \
  -e TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NCCL_P2P_DISABLE=0 \
  --shm-size 32g \
  sglang:blackwell-final-v2 \
  --model-path Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-total-tokens 8192 \
  --max-prefill-tokens 4096 \
  --chunked-prefill-size 2048 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --attention-backend triton \
  --sampling-backend pytorch \
  --schedule-policy lof \
  --schedule-conservativeness 0.95 \
  --enable-torch-compile \
  --torch-compile-max-bs 8 \
  --num-continuous-decode-steps 2 \
  --triton-attention-reduce-in-fp32 \
  --triton-attention-num-kv-splits 8 \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-overlap-schedule \
  --decode-log-interval 30 \
  --stream-output \
  --stream-interval 1 \
  --allow-auto-truncate

echo ""
echo "✅ Qwen3-8B container started"
echo ""
echo "📊 Optimized Settings:"
echo "  - Model: Qwen3-8B (8B parameters)"
echo "  - Max Tokens: 8192"
echo "  - Torch Compile: Batch size 8"
echo "  - Schedule: LOF (Least Outstanding First)"
echo "  - Continuous Decode: 2 steps"
echo "  - Triton KV Splits: 8"
echo "  - Memory: 85% static allocation"
echo ""
echo "🔗 API Endpoint: http://localhost:8000"
echo ""
echo "📝 Check status: docker logs -f qwen3-8b-sglang"
echo "⏰ Model loading may take 1-2 minutes..."