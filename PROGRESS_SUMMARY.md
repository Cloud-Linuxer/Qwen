# SGLang RTX 5090 배포 작업 요약

## 🎯 목표
- **모델**: Qwen3-32B-AWQ (4-bit quantized)
- **프레임워크**: SGLang (vLLM 대안)
- **하드웨어**: RTX 5090 (Blackwell, 32GB VRAM)
- **요구사항**: CPU 오프로딩 없이 순수 GPU 실행

## 📊 진행 상태

| 시도 | 방법 | 상태 | 문제점 |
|-----|------|------|--------|
| 1 | PyTorch 2.7.0 + 소스 빌드 | ❌ 실패 | ABI 불일치, libnuma 누락 |
| 2 | libnuma 추가 + 소스 빌드 | ❌ 실패 | Segmentation fault |
| 3 | 최적화 기능 비활성화 | ❌ 실패 | 여전히 segfault |
| 4 | **Pre-built Blackwell Wheel** | 🔄 진행 중 | PyTorch nightly 다운로드 중 |

## 🔑 핵심 인사이트

### 작동하지 않는 것들
- ❌ FlashInfer (Blackwell 미지원)
- ❌ CUDA graphs (sm_120 비호환)
- ❌ Custom kernels (ABI 불일치)
- ❌ Radix cache (메모리 충돌)

### 필요한 조건
- ✅ PyTorch nightly (2.10.0.dev)
- ✅ Pre-built Blackwell wheel
- ✅ torch_native attention backend
- ✅ 모든 최적화 비활성화

## 🚀 현재 시도 (시도 4)

### Dockerfile 구조
```
1. nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
2. PyTorch nightly 설치
3. SGLang[all] 설치
4. Blackwell 전용 sgl_kernel wheel 설치
```

### 예상 소요 시간
- 빌드: ~10-15분
- 모델 다운로드: ~5분 (캐시됨)
- 서버 시작: ~2분

## 📈 다음 단계

1. **현재 빌드 완료 대기**
2. **테스트 실행**
   - 헬스체크
   - 간단한 추론
   - 성능 측정

3. **성공 시**
   - 벤치마크 실행
   - 프로덕션 배포 준비

4. **실패 시**
   - vLLM 대안 시도
   - SGLang GitHub 이슈 제출

## 🎯 최종 목표
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-32B-AWQ", "prompt": "Hello", "max_tokens": 100}'
```
위 명령이 정상 작동하는 것!

---
*마지막 업데이트: 2025-09-16 00:39*