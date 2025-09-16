# SGLang RTX 5090 Deployment Guide

## 🚀 Qwen3-32B-AWQ on RTX 5090 with SGLang

This repository contains the complete solution for deploying Qwen3-32B-AWQ model on NVIDIA RTX 5090 (Blackwell architecture) using SGLang, achieving optimal performance without CPU offloading.

## 📊 Performance Results

### Final Optimized Performance
- **Throughput**: 10.2 tokens/s (27% improvement)
- **Short Response Latency**: ~1049ms average
- **Time to First Token (TTFT)**: 283ms
- **Korean Processing**: 9.61 tokens/s
- **GPU Usage**: 20.9GB / 32GB VRAM
- **Power**: 480W, Temperature: 57°C stable

### Performance Comparison
| Configuration | Throughput | Latency | Notes |
|--------------|-----------|---------|--------|
| Baseline (torch_native) | 8.0 tok/s | 1250ms | Initial deployment |
| Triton Optimized | 10.2 tok/s | 1049ms | **Final configuration** ✅ |
| CUDA Graphs (attempted) | Failed | N/A | Incompatible with Blackwell |
| FlashInfer (attempted) | Failed | N/A | Segmentation fault |

## 🛠️ Quick Start

### Prerequisites
- NVIDIA RTX 5090 (32GB VRAM)
- Docker with NVIDIA runtime
- CUDA 12.8+ drivers
- 50GB+ free disk space

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/qwen-sglang-rtx5090.git
cd qwen-sglang-rtx5090
```

### 2. Build Docker Image
```bash
docker build -f Dockerfile.blackwell-final -t sglang:blackwell-final-v2 .
```

### 3. Deploy SGLang Server
```bash
./deploy-sglang-final-optimized.sh
```

### 4. Test Deployment
```bash
# Quick test
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Hello, how are you?",
    "max_tokens": 20
  }'

# Run comprehensive tests
./test_sglang_commands.sh

# Run performance benchmark
./performance_test_final.sh
```

## 📁 Repository Structure

```
/home/qwen/
├── Dockerfile.blackwell-final      # Working Docker image for RTX 5090
├── deploy-sglang-final-optimized.sh # Production deployment script
├── performance_test_final.sh       # Comprehensive benchmark suite
├── test_sglang_commands.sh        # API testing commands
├── benchmark_sglang.sh             # Quick performance test
├── SGLANG_RTX5090_TRIAL_LOG.md    # Complete troubleshooting history
└── README.md                       # This file
```

## 🔧 Configuration Details

### Working Dockerfile (Dockerfile.blackwell-final)
```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Critical: libnuma libraries required for SGLang
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git build-essential \
    libnuma-dev libnuma1 && \
    rm -rf /var/lib/apt/lists/*

# PyTorch nightly for Blackwell support
RUN pip3 install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# SGLang with all dependencies
RUN pip3 install "sglang[all]"

# Pre-built CUDA 12.8 kernel (not Blackwell-specific)
RUN pip3 install --no-cache-dir \
  https://github.com/sgl-project/whl/releases/download/v0.3.9.post2/sgl_kernel-0.3.9.post2%2Bcu128-cp310-abi3-manylinux2014_x86_64.whl

ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
```

### Optimized Deployment Configuration
```bash
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
  --attention-backend triton \      # ✅ Best performance
  --sampling-backend pytorch \
  --disable-cuda-graph \            # ⚠️ Required for Blackwell
  --disable-custom-all-reduce \     # ⚠️ Required for stability
  --disable-flashinfer \            # ⚠️ Incompatible
  --disable-radix-cache \
  --decode-log-interval 40 \
  --stream-output
```

## 🎯 Key Optimizations Applied

### ✅ Working Optimizations
1. **Triton Attention Backend**: 27% performance improvement
2. **CUDA_LAUNCH_BLOCKING=0**: Enables async execution
3. **Increased Token Limits**: 3072 max tokens for better throughput
4. **Memory Fraction 0.85**: Optimal VRAM utilization
5. **PyTorch Sampling**: More stable than default
6. **Streaming Output**: Better user experience

### ❌ Incompatible with RTX 5090 (Blackwell)
- **CUDA Graphs**: Causes segmentation faults
- **FlashInfer**: Not compatible with sm_120
- **Custom All-Reduce**: Stability issues
- **AWQ_Marlin**: OOM errors with 32GB VRAM
- **Radix Cache**: Performance degradation

## 🐛 Troubleshooting Guide

### Issue 1: libnuma.so.1 Missing
**Error**: `ImportError: libnuma.so.1: cannot open shared object file`
**Solution**: Install libnuma libraries
```bash
apt-get install libnuma-dev libnuma1
```

### Issue 2: Out of Memory
**Error**: `torch.OutOfMemoryError: CUDA out of memory`
**Solutions**:
1. Reduce max-total-tokens to 2048
2. Lower mem-fraction-static to 0.8
3. Use regular AWQ instead of AWQ_Marlin

### Issue 3: Segmentation Fault
**Error**: `Segmentation fault (core dumped)`
**Solution**: Disable incompatible features
```bash
--disable-cuda-graph \
--disable-flashinfer \
--disable-custom-all-reduce
```

### Issue 4: Slow Performance
**Expected Performance by Token Count**:
- 10 tokens: ~1 second
- 50 tokens: ~5 seconds
- 100 tokens: ~10 seconds

**If slower, check**:
1. Triton attention backend is enabled
2. CUDA_LAUNCH_BLOCKING=0 is set
3. No CPU offloading is occurring
4. GPU utilization with `nvidia-smi`

## 📈 Performance Testing

### Quick Test
```bash
./benchmark_sglang.sh
```

### Comprehensive Benchmark
```bash
./performance_test_final.sh
```

### Continuous Monitoring
```bash
# GPU stats
watch -n 1 nvidia-smi

# Container logs
docker logs -f sglang-final

# API health check
while true; do
  curl -s http://localhost:8000/health | jq .
  sleep 5
done
```

## 🔬 Technical Details

### RTX 5090 Blackwell Architecture
- **CUDA Capability**: sm_120 (new Blackwell)
- **VRAM**: 32GB GDDR7
- **Memory Bandwidth**: 1.5TB/s
- **Tensor Cores**: 5th generation
- **Power**: 450W TDP

### Qwen3-32B-AWQ Model
- **Parameters**: 32 billion
- **Quantization**: 4-bit AWQ
- **Model Size**: ~16GB (quantized)
- **Context Length**: 3072 tokens (configured)
- **Vocabulary**: 152064 tokens

### SGLang vs vLLM
| Feature | SGLang | vLLM |
|---------|--------|------|
| Blackwell Support | ✅ Works with workarounds | ❌ CPU offloading required |
| Performance | 10.2 tok/s | 8-9 tok/s (with offloading) |
| Memory Usage | 20.9GB VRAM | 30GB+ with offloading |
| Stability | Stable with configs | OOM issues |
| Setup Complexity | Moderate | Simple but limited |

## 📝 Development History

### Attempts That Failed
1. **PyTorch 2.7.0 from source**: ABI mismatch issues
2. **Blackwell-specific wheel**: Doesn't exist (404)
3. **CUDA Graphs optimization**: Segmentation faults
4. **FlashInfer backend**: Incompatible with sm_120
5. **AWQ_Marlin quantization**: OOM with 32GB
6. **vLLM without offloading**: Requires CPU offload

### Critical Discovery
The missing `libnuma` libraries were causing import failures. Adding `libnuma-dev` and `libnuma1` to the Docker image resolved the core issue.

## 🚦 API Endpoints

### Completions API
```bash
POST http://localhost:8000/v1/completions
{
  "model": "Qwen/Qwen3-32B-AWQ",
  "prompt": "Your prompt here",
  "max_tokens": 100,
  "temperature": 0.7
}
```

### Chat Completions API
```bash
POST http://localhost:8000/v1/chat/completions
{
  "model": "Qwen/Qwen3-32B-AWQ",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100
}
```

### Streaming
```bash
curl -N -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "Tell me a story",
    "max_tokens": 100,
    "stream": true
  }'
```

## 📜 License
MIT

## 🤝 Contributing
Contributions welcome! Please test on RTX 5090 hardware before submitting PRs.

## 📞 Support
For issues specific to RTX 5090 deployment, please open an issue with:
1. Full error logs
2. `nvidia-smi` output
3. Docker logs
4. Configuration used

---

**Last Updated**: September 16, 2025
**Tested On**: NVIDIA RTX 5090 32GB, CUDA 12.8, Ubuntu 22.04
**Status**: ✅ Production Ready