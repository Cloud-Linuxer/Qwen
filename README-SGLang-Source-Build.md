# SGLang Source Build for RTX 5090 & PyTorch 2.7.0

This system builds SGLang and sgl_kernel from source to resolve PyTorch 2.7.0 symbol compatibility issues on RTX 5090 (Blackwell architecture).

## Problem Solved

**Issue**: `undefined symbol: _ZN3c104cuda9SetDeviceEab` when importing sgl_kernel with PyTorch 2.7.0
**Root Cause**: Prebuilt sgl_kernel packages were compiled against older PyTorch versions with different ABI
**Solution**: Build sgl_kernel from source against the exact PyTorch 2.7.0 installation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   SGLang Source Build System               │
├─────────────────────────────────────────────────────────────┤
│ 1. Base Environment (CUDA 12.8 + Ubuntu 22.04)           │
│ 2. PyTorch 2.7.0 Installation (establishes ABI)          │
│ 3. Source Code Clone (SGLang + submodules)               │
│ 4. sgl_kernel Compilation (against PyTorch 2.7.0)        │
│ 5. SGLang Installation (with custom sgl_kernel)          │
│ 6. RTX 5090 Optimization (Blackwell sm_120 support)      │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. **Dockerfile.sglang-source-build**
- Multi-stage build optimized for RTX 5090
- PyTorch 2.7.0 + CUDA 12.8 foundation
- Source compilation of sgl_kernel with proper linking
- Blackwell architecture support (sm_120)

### 2. **Build Scripts**
- `build-sglang-rtx5090.sh`: Orchestrates the entire build process
- `deploy-sglang-source-rtx5090.sh`: Deploys Qwen3-32B-AWQ with optimized settings
- `debug-sglang-build.sh`: Comprehensive debugging and validation
- `test-sglang-api.sh`: API testing and performance validation

## Build Process

### Phase 1: Environment Setup
```dockerfile
FROM nvidia/cuda:12.8-devel-ubuntu22.04
# Install Python 3.11, build tools, CUDA development toolkit
```

### Phase 2: PyTorch Foundation
```dockerfile
RUN pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# This establishes the exact ABI that sgl_kernel must match
```

### Phase 3: Source Compilation
```dockerfile
RUN cd sglang/python/sglang/srt/layers && \
    python setup.py clean --all && \
    python setup.py build_ext --inplace --force
# Compiles sgl_kernel against the installed PyTorch 2.7.0
```

### Phase 4: RTX 5090 Optimization
```dockerfile
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"
ENV CUDA_ARCHITECTURES="80;86;89;90;120"
# Ensures Blackwell (sm_120) support
```

## Quick Start

### 1. Build the Source Image
```bash
./build-sglang-rtx5090.sh
```
**Time**: 15-30 minutes
**Disk**: ~20GB required
**Output**: `sglang:rtx5090-source` Docker image

### 2. Deploy Qwen3-32B-AWQ
```bash
./deploy-sglang-source-rtx5090.sh
```
**Memory**: ~22GB VRAM for Qwen3-32B-AWQ
**Startup**: 2-5 minutes for model loading
**Endpoints**: http://localhost:8000

### 3. Test the API
```bash
./test-sglang-api.sh
```
**Coverage**: Health, models, completion, chat, performance
**Validation**: Symbol compatibility, GPU utilization

## RTX 5090 Optimizations

### Memory Management
- **Static Fraction**: 0.95 (uses 30.4GB of 32GB VRAM)
- **Dynamic Allocation**: `expandable_segments:True`
- **KV Cache**: FP8 precision for memory efficiency

### Performance Settings
- **Quantization**: AWQ-Marlin for optimal inference speed
- **Prefill Chunking**: 2048 tokens for balanced throughput
- **Torch Compile**: Enabled for graph optimization
- **Custom All-Reduce**: Disabled for Blackwell stability

### Architecture Flags
```bash
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"
CUDA_ARCHITECTURES="80;86;89;90;120"
```

## Troubleshooting

### Build Failures

**Out of Memory During Build**:
```bash
# Reduce parallel jobs
docker build --build-arg MAX_JOBS=4 ...
```

**CUDA Toolkit Issues**:
```bash
# Verify NVIDIA driver
nvidia-smi
# Check CUDA installation in container
docker run --rm nvidia/cuda:12.8-devel-ubuntu22.04 nvcc --version
```

### Runtime Issues

**Model Loading Failures**:
```bash
# Check available GPU memory
nvidia-smi
# Reduce memory fraction
# Edit docker-compose.sglang-source.yml: --mem-fraction-static 0.90
```

**Import Errors**:
```bash
# Debug specific imports
./debug-sglang-build.sh
# Check symbol compatibility
docker exec -it qwen3-32b-awq-source python -c "from sglang.srt.layers import sgl_kernel; print('OK')"
```

### Performance Issues

**Slow Inference**:
```bash
# Check GPU utilization
nvidia-smi -l 1
# Verify torch compile
docker logs qwen3-32b-awq-source | grep "torch.compile"
```

**High Memory Usage**:
```bash
# Monitor memory patterns
docker exec -it qwen3-32b-awq-source python -c "import torch; print(f'Allocated: {torch.cuda.memory_allocated(0)/1024/1024:.1f}MB')"
```

## Validation Commands

### Build Validation
```bash
# Check image exists
docker images sglang:rtx5090-source

# Verify PyTorch compatibility
docker run --rm --gpus all sglang:rtx5090-source python -c "
import torch;
from sglang.srt.layers import sgl_kernel;
print('✅ Build successful')
"
```

### Runtime Validation
```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/v1/models

# Quick test
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-32B-AWQ","messages":[{"role":"user","content":"Hello!"}],"max_tokens":10}'
```

## Alternative Deployment Methods

### Docker Compose
```bash
docker-compose -f docker-compose.sglang-source.yml up -d
```

### Manual Docker Run
```bash
docker run -d --name qwen3-32b-awq-source \
  --runtime nvidia --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  sglang:rtx5090-source \
  --model-path Qwen/Qwen3-32B-AWQ \
  --quantization awq_marlin \
  --max-total-tokens 32768
```

## Key Differences from Prebuilt

| Aspect | Prebuilt SGLang | Source Build |
|--------|----------------|--------------|
| **Compatibility** | PyTorch 2.5.x ABI | PyTorch 2.7.0 ABI |
| **Architecture** | Limited CUDA archs | Full Blackwell support |
| **Build Time** | ~2 minutes | ~20 minutes |
| **Symbol Issues** | ❌ Fails | ✅ Resolved |
| **Customization** | Limited | Full control |

## Performance Expectations

### Qwen3-32B-AWQ on RTX 5090
- **Loading Time**: 2-3 minutes
- **Memory Usage**: ~22GB VRAM
- **Throughput**: ~50-80 tokens/second (varies by sequence length)
- **Latency**: ~100-200ms first token

### Resource Requirements
- **GPU Memory**: 24GB+ (RTX 5090 recommended)
- **System RAM**: 32GB+ recommended
- **Disk Space**: 50GB+ for model cache
- **Build Space**: 20GB+ for Docker layers

## Security Considerations

- API key protection (default: `sk-sglang-key-12345`)
- Container isolation with non-root user
- Network security (bind to localhost by default)
- Model weight validation through HuggingFace

## Advanced Configuration

### Custom Model Loading
```bash
# Edit deployment script model path
--model-path /models/your-custom-model
```

### Memory Optimization
```bash
# For smaller GPUs
--mem-fraction-static 0.80
--max-total-tokens 16384
```

### Performance Tuning
```bash
# Enable additional optimizations
--enable-flashinfer  # If compatible
--enable-triton      # For custom kernels
--chunked-prefill-size 4096  # Larger chunks
```

This source build system ensures complete compatibility between SGLang, sgl_kernel, and PyTorch 2.7.0 on RTX 5090 hardware while providing optimal performance for large language model inference.