# SGLang RTX 5090 최종 성능 보고서

## 📊 성능 테스트 결과 (CSV 데이터 기반)

### 테스트 환경
- **하드웨어**: NVIDIA RTX 5090 (32GB VRAM)
- **모델**: Qwen3-32B-AWQ (4-bit 양자화)
- **프레임워크**: SGLang v0.3.9.post2
- **테스트 일시**: 2025년 9월 16일

### 성능 측정 결과

| Configuration | Avg Latency | Min Latency | Max Latency | Throughput | TTFT | Samples |
|--------------|-------------|-------------|-------------|------------|------|---------|
| **Baseline-Triton** | 1054.1ms | 980.7ms | 1059.0ms | 10.16 tok/s | 269.0ms | 20 |
| **Balanced-v2-LOF** | 975.7ms | 970.7ms | 977.1ms | 10.27 tok/s | 231.5ms | 10 |

### 성능 개선율

| 지표 | Baseline | Balanced-v2 | 개선율 |
|------|----------|-------------|--------|
| **평균 응답 지연** | 1054.1ms | 975.7ms | **+7.4%** ✅ |
| **처리량** | 10.16 tok/s | 10.27 tok/s | **+1.1%** ✅ |
| **첫 토큰 시간 (TTFT)** | 269.0ms | 231.5ms | **+13.9%** ✅ |

## 🎯 핵심 성능 개선 내역

### 1. 응답 지연 개선: 7.4%
- **Before**: 1054.1ms (Baseline)
- **After**: 975.7ms (Balanced-v2)
- **절대 개선**: 78.4ms 단축
- **일관성 개선**: 최대-최소 편차 78.3ms → 6.4ms (92% 감소)

### 2. Time to First Token (TTFT) 개선: 13.9%
- **Before**: 269.0ms
- **After**: 231.5ms
- **절대 개선**: 37.5ms 단축
- **사용자 체감**: 스트리밍 응답 시작이 눈에 띄게 빨라짐

### 3. 처리량 개선: 1.1%
- **Before**: 10.16 tok/s
- **After**: 10.27 tok/s
- **절대 개선**: 0.11 tok/s 증가
- **안정성**: 일관된 처리 속도 유지

## 🔧 적용된 최적화 기법

### Balanced-v2 구성 (최종 최적화)
```bash
--schedule-policy lof          # Least Outstanding First 스케줄링
--enable-torch-compile         # PyTorch JIT 컴파일 활성화
--torch-compile-max-bs 6       # 배치 크기 6까지 컴파일
--num-continuous-decode-steps 2 # 연속 디코드 2 스텝
--triton-attention-num-kv-splits 12 # KV 캐시 12분할
--mem-fraction-static 0.87     # 메모리 87% 정적 할당
```

### 주요 최적화 효과
1. **LOF 스케줄링**: 요청 우선순위 최적화로 지연 감소
2. **Torch Compile**: 반복 실행 시 추가 최적화
3. **연속 디코드**: 배치 처리 효율 증가
4. **메모리 최적화**: 안정성과 성능의 균형

## 📈 전체 성능 향상 (vLLM → SGLang)

| 단계 | 처리량 | 누적 개선 |
|------|--------|----------|
| vLLM (CPU offload) | ~8 tok/s | Baseline |
| SGLang (초기) | 10.2 tok/s | +27.5% |
| SGLang (Balanced-v2) | 10.27 tok/s | **+28.4%** |

## 💡 주요 발견사항

### 효과적인 최적화
✅ LOF (Least Outstanding First) 스케줄링 정책
✅ 적절한 연속 디코드 스텝 (2가 최적)
✅ Torch Compile (배치 크기 6)
✅ Triton Attention Backend

### 비효과적/문제 있는 최적화
❌ CUDA Graphs (Blackwell 비호환)
❌ FlashInfer (sm_120 지원 없음)
❌ FP8 KV Cache (OOM 발생)
❌ 과도한 연속 디코드 스텝 (3+)

## 🏁 결론

SGLang Balanced-v2 최적화를 통해:
- **응답 속도 7.4% 개선**
- **TTFT 13.9% 개선**
- **안정적인 처리량 유지**
- **일관성 있는 성능** (편차 92% 감소)

RTX 5090 Blackwell 아키텍처에서 Qwen3-32B-AWQ 모델을 위한 최적 구성을 달성했습니다.

## 📁 테스트 데이터
- `comparison_20250916_154056.csv` - 최종 비교 데이터
- `final_benchmark_20250916_153814.csv` - 상세 벤치마크 결과