#!/bin/bash

# SGLang 균형 최적화 v2
# 안정성과 성능의 균형

echo "⚖️ SGLang 균형 최적화 v2 배포..."
echo "📋 균형 최적화:"
echo "  ✅ Torch Compile (안정적 설정)"
echo "  ✅ 스마트 스케줄링 (LOF)"
echo "  ✅ 중간 연속 디코드"
echo "  ✅ 최적 메모리 설정"
echo "  ✅ 개선된 Triton 설정"

# 기존 컨테이너 정리
docker stop sglang-balanced-v2 2>/dev/null
docker rm sglang-balanced-v2 2>/dev/null

# 균형 최적화 배포
docker run -d \
  --name sglang-balanced-v2 \
  --runtime nvidia \
  --gpus all \
  -p 8003:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256" \
  -e CUDA_LAUNCH_BLOCKING=0 \
  -e OMP_NUM_THREADS=10 \
  -e TORCH_CUDA_ARCH_LIST="9.0;12.0+PTX" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NCCL_P2P_DISABLE=0 \
  --shm-size 40g \
  sglang:blackwell-final-v2 \
  --model-path Qwen/Qwen3-32B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --max-total-tokens 3328 \
  --max-prefill-tokens 1664 \
  --chunked-prefill-size 1152 \
  --mem-fraction-static 0.87 \
  --trust-remote-code \
  --attention-backend triton \
  --sampling-backend pytorch \
  --schedule-policy lof \
  --schedule-conservativeness 0.95 \
  --enable-torch-compile \
  --torch-compile-max-bs 6 \
  --num-continuous-decode-steps 2 \
  --triton-attention-reduce-in-fp32 \
  --triton-attention-num-kv-splits 12 \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-overlap-schedule \
  --decode-log-interval 30 \
  --stream-output \
  --stream-interval 1 \
  --allow-auto-truncate

echo "✅ 균형 최적화 v2 컨테이너 시작됨"
echo ""
echo "⚖️ 균형 설정:"
echo "  - Torch Compile: 배치 크기 6"
echo "  - Schedule: LOF (Least Outstanding First)"
echo "  - 연속 디코드: 2 스텝"
echo "  - Triton KV 분할: 12"
echo "  - 자동 트렁케이트: 활성화"
echo "  - 메모리: 87% 정적 할당"
echo ""
echo "🔗 API 엔드포인트: http://localhost:8003"