# SGLang RTX 5090 배포 시행착오 기록

## 환경
- GPU: NVIDIA RTX 5090 (Blackwell, sm_120)
- CUDA: 12.8
- OS: Ubuntu 22.04
- 목표: Qwen3-32B-AWQ 모델을 SGLang으로 서빙

---

## 시도 1: PyTorch 2.7.0 + 소스 빌드
**시간**: 2025-09-16 00:00~00:20

### Dockerfile
```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
RUN pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
# SGLang 소스에서 빌드
RUN git clone https://github.com/sgl-project/sglang.git && cd sglang && pip install -e .
```

### 결과
❌ **실패**
- sgl_kernel 빌드 중 ABI 불일치
- libnuma.so.1 누락 에러

### 학습
- PyTorch와 sgl_kernel 간 ABI 호환성 중요
- 시스템 라이브러리 의존성 확인 필요

---

## 시도 2: libnuma 추가 + 소스 빌드
**시간**: 2025-09-16 00:20~00:30

### 변경사항
```dockerfile
RUN apt-get install -y libnuma-dev libnuma1
```

### 결과
❌ **실패**
- 빌드는 성공 (exit code 0)
- 런타임 Segmentation fault
- flashinfer backend 충돌

### 학습
- Blackwell 아키텍처와 flashinfer 비호환
- CUDA graph 초기화 실패

---

## 시도 3: 최적화 기능 비활성화
**시간**: 2025-09-16 00:30~00:35

### 실행 명령
```bash
docker run ... \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native
```

### 결과
❌ **실패**
- 여전히 Segmentation fault
- warmup 단계에서 충돌
- TorchInductor compilation 에러

### 학습
- 단순 옵션 비활성화로는 해결 안됨
- 더 근본적인 호환성 문제 존재

---

## 시도 4: Pre-built Blackwell Wheel (실패)
**시간**: 2025-09-16 00:37~01:10

### 첫 번째 접근
- PyTorch nightly + Blackwell 전용 wheel
- **문제**: Blackwell 전용 wheel이 존재하지 않음 (404 에러)
- SGLang이 PyTorch 2.8.0으로 다운그레이드

---

## 시도 5: 일반 CUDA 12.8 Wheel
**시간**: 2025-09-16 01:10~01:15

### 수정된 접근
```dockerfile
# PyTorch nightly (최신 Blackwell 지원)
RUN pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# SGLang 설치 (PyTorch 2.8.0으로 다운그레이드됨)
RUN pip3 install "sglang[all]"

# 일반 CUDA 12.8 wheel (v0.3.9.post2) - Blackwell 전용 없음
RUN pip3 install https://github.com/sgl-project/whl/releases/download/v0.3.9.post2/sgl_kernel-0.3.9.post2%2Bcu128-cp310-abi3-manylinux2014_x86_64.whl
```

### 결과
❌ **실패**
- 빌드는 성공했지만 libnuma 누락 에러
- ImportError: libnuma.so.1: cannot open shared object file

---

## 시도 6: libnuma 추가 후 재빌드 ✅ **성공**
**시간**: 2025-09-16 01:15~01:20

### 수정사항
```dockerfile
RUN apt-get install -y libnuma-dev libnuma1
```

### 차이점
1. **PyTorch nightly 사용** - 최신 Blackwell 지원
2. **Pre-built wheel 사용** - 소스 빌드 복잡성 회피
3. **libnuma 라이브러리 추가** - sgl_kernel 의존성 해결

### 빌드 로그
- PyTorch 2.10.0.dev20250915+cu128 다운로드 중 (901.7 MB)
- Python 3.10 사용 (이전 3.11과 다름)
- nvidia-cudnn-cu12==9.10.2.21 (최신 버전)

### 최종 결과 - 시도 6 (01:15~01:20)
- ✅ libnuma-dev, libnuma1 패키지 설치
- ✅ PyTorch 2.10.0.dev20250915+cu128 설치
- ✅ SGLang[all] 설치 (torch 2.8.0으로 다운그레이드)
- ✅ sgl_kernel v0.3.9.post2 cu128 wheel 설치
- ✅ **서버 시작 성공!**
- ✅ **API 테스트 성공!** - Qwen3-32B-AWQ 모델 정상 동작

---

## 중요 발견사항

### PyTorch 버전 진화
- **2.7.0**: Blackwell 초기 지원
- **2.8.0**: 개선되었지만 SGLang과 충돌
- **2.10.0.dev (nightly)**: 최신 Blackwell 최적화

### Python 버전 영향
- Python 3.10: 기본 Ubuntu 22.04 버전, 호환성 좋음
- Python 3.11: 수동 설치 필요, 일부 패키지와 충돌 가능

### Pre-built vs Source Build
| 방식 | 장점 | 단점 |
|------|------|------|
| Source Build | 커스터마이징 가능 | ABI 불일치, 빌드 복잡 |
| Pre-built Wheel | 테스트 완료, 간단 | 유연성 부족 |

---

## 핵심 문제점 정리

### 1. Blackwell 아키텍처 미지원
- sgl_kernel이 sm_120을 완전히 지원하지 않음
- FlashInfer, CUDA graphs 등 최적화 기능 비호환

### 2. PyTorch 버전 문제
- 2.7.0: Blackwell 부분 지원
- 2.8.0+: 개선된 지원 (하지만 SGLang과 충돌)
- Nightly: 최신 패치 포함

### 3. 컴파일 vs Pre-built
- 소스 컴파일: ABI 불일치, 복잡한 빌드 과정
- Pre-built wheel: 테스트된 바이너리, 안정성

---

## 대안 검토

### vLLM
- 더 성숙한 프로젝트
- RTX 5090 지원 개선 중
- 하지만 여전히 완벽하지 않음

### 권장사항
1. SGLang 팀의 공식 Blackwell 지원 대기
2. PyTorch 2.8.x 안정화 대기
3. 당분간 CPU offloading 사용 고려

---

## 다음 단계
1. Pre-built Blackwell wheel 테스트 완료
2. 성공 시 → 성능 벤치마크
3. 실패 시 → vLLM 대안 시도

---

## 🎉 최종 해결 방법

### 작동하는 Dockerfile
```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git build-essential \
    libnuma-dev libnuma1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install --pre torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/nightly/cu128

RUN pip3 install "sglang[all]"

RUN pip3 install --no-cache-dir \
  https://github.com/sgl-project/whl/releases/download/v0.3.9.post2/sgl_kernel-0.3.9.post2%2Bcu128-cp310-abi3-manylinux2014_x86_64.whl

ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

WORKDIR /workspace
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
```

### 배포 명령
```bash
docker run -d \
  --name sglang-blackwell \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 16g \
  sglang:blackwell-final-v2 \
  --model-path Qwen/Qwen3-32B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --max-total-tokens 2048 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --disable-flashinfer \
  --disable-radix-cache \
  --attention-backend torch_native
```

### 핵심 포인트
1. **libnuma 필수**: sgl_kernel이 libnuma.so.1에 의존
2. **PyTorch nightly 필요**: Blackwell 아키텍처 지원
3. **Pre-built wheel 사용**: 소스 빌드 대신 공식 wheel 사용
4. **모든 최적화 비활성화**: Blackwell과 호환성 문제 회피

---

*문서 완료: 2025-09-16 01:20*