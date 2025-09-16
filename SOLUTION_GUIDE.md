# SGLang RTX 5090 해결 가이드

## 문제 진단
현재 SGLang + RTX 5090 조합의 주요 문제:
1. sgl_kernel과 PyTorch 2.7.0 ABI 불일치
2. FlashInfer가 Blackwell 아키텍처와 충돌
3. CUDA graph 초기화 시 segfault

## 해결 방법

### 옵션 1: 안정적 빌드 (권장)
```bash
# 1. 안정적인 이미지 빌드
./deploy-sglang-stable.sh

# 2. 로그 확인
docker logs -f sglang-stable

# 3. 문제 발생시 디버깅
docker exec -it sglang-stable bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 옵션 2: vLLM 대체
```bash
# SGLang이 계속 실패하면
./deploy-vllm-alternative.sh

# vLLM은 더 안정적이고 RTX 5090 지원 개선됨
docker logs -f vllm-rtx5090
```

### 옵션 3: 점진적 디버깅
```bash
# 1. 최소 설정으로 시작
docker run --rm -it \
  --runtime nvidia \
  --gpus all \
  sglang:stable \
  python -c "import sgl_kernel; print('OK')"

# 2. 단계별 기능 추가
# - 먼저 모델 로딩만
# - 그 다음 간단한 추론
# - 마지막으로 전체 서버
```

## 핵심 해결 포인트

### ✅ DO (해야 할 것)
1. **PyTorch 2.7.0 정식 버전 사용** (nightly 금지)
2. **xformers attention 백엔드 사용**
3. **메모리 설정 보수적으로** (85% 이하)
4. **CUDA_LAUNCH_BLOCKING=1로 디버깅**
5. **단계별 검증** (import → 모델 로드 → 추론)

### ❌ DON'T (하지 말아야 할 것)
1. **FlashInfer 사용 금지** (Blackwell 미지원)
2. **CUDA graph 활성화 금지** (segfault 원인)
3. **radix cache 사용 금지** (메모리 충돌)
4. **nightly 빌드 사용 금지** (불안정)

## 검증 체크리스트

```bash
# 1. GPU 인식 확인
nvidia-smi

# 2. PyTorch CUDA 확인
docker exec sglang-stable python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# 3. sgl_kernel 확인
docker exec sglang-stable python -c "
import sgl_kernel
print('sgl_kernel loaded')
"

# 4. 모델 로딩 테스트
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Test",
    "max_tokens": 10
  }'
```

## 예상 소요 시간
- 이미지 빌드: 15-20분
- 모델 다운로드: 10-15분 (캐시된 경우 skip)
- 서버 시작: 2-3분

## 문제 지속시
1. vLLM으로 전환 (더 안정적)
2. SGLang GitHub에 RTX 5090 이슈 리포트
3. PyTorch 2.8.0 출시 대기 (Blackwell 개선 예정)