# SGLang RTX 5090 트러블슈팅 차트

## 에러별 해결 방법

### 🔴 Segmentation Fault
**증상**: `Fatal Python error: Segmentation fault`

**원인**:
- FlashInfer가 Blackwell과 비호환
- CUDA graph 초기화 실패
- sgl_kernel ABI 불일치

**해결책**:
```bash
--disable-flashinfer
--disable-cuda-graph
--attention-backend torch_native
```

### 🟡 ABI Incompatibility
**증상**: `undefined symbol: _ZN3c10...`

**원인**:
- PyTorch와 sgl_kernel 컴파일 버전 불일치
- C++ ABI 버전 차이

**해결책**:
- Pre-built wheel 사용
- 동일한 PyTorch 버전으로 재컴파일

### 🟠 Missing Libraries
**증상**: `libnuma.so.1: cannot open shared object file`

**원인**:
- 시스템 라이브러리 누락

**해결책**:
```dockerfile
RUN apt-get install -y libnuma-dev libnuma1
```

### 🔵 CUDA Kernel Error
**증상**: `no kernel image is available for execution on the device`

**원인**:
- sm_120 (Blackwell) 미지원

**해결책**:
- PyTorch nightly 사용
- TORCH_CUDA_ARCH_LIST="12.0+PTX" 설정

### 🟢 HTTP 503 Error
**증상**: Health check returns 503

**원인**:
- 서버 초기화 실패
- 모델 로딩 실패

**해결책**:
- 메모리 설정 조정 (--mem-fraction-static 0.65)
- 토큰 수 제한 (--max-total-tokens 1024)

## 버전 호환성 매트릭스

| PyTorch | CUDA | sgl_kernel | RTX 5090 지원 | 상태 |
|---------|------|------------|--------------|------|
| 2.7.0 | 12.8 | Source build | 부분 | ❌ Segfault |
| 2.8.0 | 12.8 | Source build | 개선 | ❌ API 변경 |
| 2.10.0.dev | 12.8 | Pre-built Blackwell | 최적 | 🔄 테스트 중 |

## 실행 명령 템플릿

### 최소 설정 (안정성 우선)
```bash
docker run -d \
  --name sglang \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e CUDA_LAUNCH_BLOCKING=1 \
  --shm-size 16g \
  sglang:blackwell \
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
  --attention-backend torch_native
```

### 디버깅 설정
```bash
# 추가 환경변수
-e TORCH_SHOW_CPP_STACKTRACES=1 \
-e TORCHINDUCTOR_COMPILE_THREADS=1 \
-e CUDA_LAUNCH_BLOCKING=1 \
--cap-add SYS_PTRACE \
--security-opt seccomp=unconfined
```

## 체크리스트

### 빌드 전
- [ ] CUDA 12.8 설치 확인
- [ ] GPU 메모리 확인 (32GB)
- [ ] 디스크 공간 확인 (>100GB)

### 빌드 중
- [ ] PyTorch nightly 사용
- [ ] Pre-built Blackwell wheel 다운로드
- [ ] Python 3.10 사용 (기본)

### 실행 전
- [ ] 기존 컨테이너 정리
- [ ] 포트 8000 사용 가능
- [ ] HuggingFace 캐시 마운트

### 실행 중
- [ ] Health check 통과
- [ ] 간단한 추론 테스트
- [ ] 메모리 사용량 모니터링

## 알려진 제한사항

1. **FlashInfer 사용 불가**
   - Blackwell 아키텍처 미지원
   - torch_native로 대체 (성능 저하)

2. **CUDA Graphs 비활성화**
   - sm_120 초기화 실패
   - 추론 속도 영향

3. **최대 토큰 수 제한**
   - 메모리 안정성 위해 1024-2048 권장
   - 프로덕션에서는 점진적 증가

4. **PyTorch Nightly 의존**
   - 안정성 보장 없음
   - 정식 버전 출시 대기 필요

---
*문서 작성: 2025-09-16 00:40*