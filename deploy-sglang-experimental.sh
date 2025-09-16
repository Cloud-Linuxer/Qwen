#!/bin/bash

# SGLang 실험적 최적화 버전
# 공격적인 최적화 설정

echo "🔬 SGLang 실험적 최적화 버전 배포..."
echo "⚠️ 경고: 실험적 설정 - 불안정할 수 있음"
echo "📋 실험적 최적화:"
echo "  ✅ CUDA 그래프 (작은 배치)"
echo "  ✅ DFS 가중치 스케줄링"
echo "  ✅ 계층적 캐시"
echo "  ✅ 혼합 청크 처리"
echo "  ✅ 두 배치 오버랩"

# 기존 컨테이너 정리
docker stop sglang-experimental 2>/dev/null
docker rm sglang-experimental 2>/dev/null

# 실험적 최적화 배포
docker run -d \
  --name sglang-experimental \
  --runtime nvidia \
  --gpus all \
  -p 8002:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64" \
  -e CUDA_LAUNCH_BLOCKING=0 \
  -e OMP_NUM_THREADS=16 \
  -e TORCH_CUDA_ARCH_LIST="12.0+PTX" \
  -e CUDA_DEVICE_ORDER="PCI_BUS_ID" \
  -e PYTORCH_ENABLE_MPS_FALLBACK=1 \
  --shm-size 48g \
  sglang:blackwell-final-v2 \
  --model-path Qwen/Qwen3-32B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --max-total-tokens 3584 \
  --max-prefill-tokens 1792 \
  --chunked-prefill-size 1280 \
  --mem-fraction-static 0.90 \
  --trust-remote-code \
  --attention-backend triton \
  --sampling-backend pytorch \
  --schedule-policy dfs-weight \
  --schedule-conservativeness 0.8 \
  --cuda-graph-max-bs 4 \
  --cuda-graph-bs 1 2 4 \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --enable-mixed-chunk \
  --enable-two-batch-overlap \
  --tbo-token-distribution-threshold 0.45 \
  --num-continuous-decode-steps 4 \
  --triton-attention-reduce-in-fp32 \
  --triton-attention-num-kv-splits 24 \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --decode-log-interval 10 \
  --stream-output \
  --stream-interval 1

echo "✅ 실험적 컨테이너 시작됨"
echo ""
echo "🔬 실험적 설정:"
echo "  - CUDA Graphs: 1,2,4 배치 크기"
echo "  - Schedule: DFS-Weight (깊이 우선 가중치)"
echo "  - 계층적 캐시: 2.0 비율"
echo "  - 혼합 청크: 활성화"
echo "  - 두 배치 오버랩: 활성화"
echo "  - 연속 디코드: 4 스텝"
echo ""
echo "🔗 API 엔드포인트: http://localhost:8002"