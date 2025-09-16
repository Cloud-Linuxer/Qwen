#!/bin/bash

# SGLang 최종 최적화 버전
# 안정적으로 작동하는 최적화만 포함

echo "🚀 SGLang 최종 최적화 버전 배포..."
echo "📋 적용 최적화:"
echo "  ✅ Triton attention backend"
echo "  ✅ CUDA 비동기 실행 (CUDA_LAUNCH_BLOCKING=0)"
echo "  ✅ 메모리 최적화 설정"
echo "  ✅ 증가된 토큰 한계 (3072)"
echo "  ✅ PyTorch 샘플링 백엔드"

# 기존 컨테이너 정리
docker stop sglang-final 2>/dev/null
docker rm sglang-final 2>/dev/null

# 최종 최적화 배포
docker run -d \
  --name sglang-final \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256" \
  -e CUDA_LAUNCH_BLOCKING=0 \
  -e OMP_NUM_THREADS=8 \
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
  --decode-log-interval 40 \
  --stream-output

echo "✅ 컨테이너 시작됨"
echo ""
echo "📊 예상 성능:"
echo "  - 처리 속도: ~10 tokens/s"
echo "  - 짧은 응답: ~1초"
echo "  - 중간 응답 (50 토큰): ~5초"
echo "  - 긴 응답 (100 토큰): ~10초"
echo ""
echo "🔗 API 엔드포인트: http://localhost:8000"
echo ""
echo "테스트 명령:"
echo "  curl -X POST http://localhost:8000/v1/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"Qwen/Qwen3-32B-AWQ\", \"prompt\": \"Hello\", \"max_tokens\": 10}'"