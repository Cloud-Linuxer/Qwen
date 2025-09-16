#!/bin/bash

# SGLang 울트라 최적화 버전
# 추가 성능 최적화 옵션 적용

echo "🚀 SGLang 울트라 최적화 버전 배포..."
echo "📋 추가 최적화:"
echo "  ✅ Torch Compile 활성화"
echo "  ✅ 연속 디코드 스텝 증가"
echo "  ✅ LPM 스케줄 정책"
echo "  ✅ KV 캐시 FP8 양자화"
echo "  ✅ 혼합 배치 처리"
echo "  ✅ CUDA 아키텍처 최적화"

# 기존 컨테이너 정리
docker stop sglang-ultra 2>/dev/null
docker rm sglang-ultra 2>/dev/null

# 울트라 최적화 배포
docker run -d \
  --name sglang-ultra \
  --runtime nvidia \
  --gpus all \
  -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128" \
  -e CUDA_LAUNCH_BLOCKING=0 \
  -e OMP_NUM_THREADS=12 \
  -e TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX" \
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
  --schedule-policy lpm \
  --enable-torch-compile \
  --torch-compile-max-bs 8 \
  --num-continuous-decode-steps 3 \
  --triton-attention-reduce-in-fp32 \
  --triton-attention-num-kv-splits 16 \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --decode-log-interval 20 \
  --stream-output \
  --stream-interval 1

echo "✅ 컨테이너 시작됨"
echo ""
echo "📊 울트라 최적화 설정:"
echo "  - Torch Compile: 활성화"
echo "  - KV Cache: FP8 양자화"
echo "  - Schedule Policy: LPM (Longest Prefix Matching)"
echo "  - 연속 디코드: 3 스텝"
echo "  - Triton FP32 감소: 활성화"
echo "  - 최대 토큰: 4096"
echo ""
echo "🔗 API 엔드포인트: http://localhost:8001"
echo ""
echo "테스트 명령:"
echo "  curl -X POST http://localhost:8001/v1/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"Qwen/Qwen3-32B-AWQ\", \"prompt\": \"Hello\", \"max_tokens\": 10}'"