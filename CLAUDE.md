# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Qwen3 LLM deployment project optimized for RTX 5090 hardware, providing a containerized infrastructure for running large language models with vLLM.

## Architecture

The system consists of three main services:
- **vLLM Server** (port 8000): Model inference server handling LLM requests
- **Backend API** (port 8080): Python API gateway for request processing
- **Frontend UI** (port 8501): Streamlit-based user interface

## Development Commands

### Deployment
```bash
# Deploy all services (recommended method)
./deploy-qwen.sh

# Alternative: Direct docker-compose
docker-compose -f docker-compose.qwen.yml up -d
```

### Service Management
```bash
# View service logs
docker-compose -f docker-compose.qwen.yml logs -f qwen3-next

# Stop all services
docker-compose -f docker-compose.qwen.yml down

# Restart services
docker-compose -f docker-compose.qwen.yml restart

# Check service status
docker-compose -f docker-compose.qwen.yml ps
```

### Monitoring
```bash
# Check GPU usage
nvidia-smi -l 1

# Health check endpoints
curl http://localhost:8000/health   # vLLM server
curl http://localhost:8080/health   # Backend API
curl http://localhost:8501/healthz  # Frontend
```

### Testing API
```bash
# Test vLLM directly
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B",
    "prompt": "Hello, ",
    "max_tokens": 10,
    "temperature": 0.7
  }'

# Test via backend API
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Key Configurations

### Memory Management
The deployment is configured for RTX 5090 with specific memory constraints:
- GPU memory utilization: 90% (~30GB of 32GB VRAM)
- CPU offloading: 40GB for model weights
- Max context length: 8192 tokens (adjustable in docker-compose.qwen.yml)

### RTX 5090 Specific Settings
Due to Blackwell architecture (sm_120), special configurations are applied:
- CUDA architecture list includes 12.0+PTX
- Custom all-reduce disabled for stability
- Triton flash attention enabled
- CUDA graphs disabled

## File Structure & Purpose

- **deploy-qwen.sh**: Main deployment script with pre-flight checks (GPU, disk space, memory)
- **docker-compose.qwen.yml**: Service orchestration and configuration
- **qwen3_next_support.py**: Patches vLLM to support Qwen3-Next architecture
- **vllm_qwen3_startup.py**: Server startup script with model registration
- **Dockerfile.qwen3-next[-patched]**: Container images for vLLM server

## Common Modifications

### Adjust Memory/Context
Edit `docker-compose.qwen.yml`:
- `--gpu-memory-utilization`: Adjust GPU memory usage (0.0-1.0)
- `--cpu-offload-gb`: Increase for larger models
- `--max-model-len`: Reduce if OOM errors occur

### Change Model
Update model name in:
1. `docker-compose.qwen.yml`: Line 41 (--model parameter)
2. `docker-compose.qwen.yml`: Lines 84-85 (VLLM_MODEL env var)

### Performance Tuning
- Enable quantization: Add `--quantization gptq` to vLLM command
- Adjust batch size: Add `--max-num-seqs` parameter
- Configure tensor parallelism: Modify `--tensor-parallel-size`

## Troubleshooting

### Out of Memory
1. Reduce `--max-model-len` in docker-compose.qwen.yml
2. Increase `--cpu-offload-gb` value
3. Lower `--gpu-memory-utilization`

### Service Won't Start
1. Check logs: `docker-compose -f docker-compose.qwen.yml logs qwen3-next`
2. Verify GPU availability: `nvidia-smi`
3. Ensure sufficient disk space: `df -h`

### Slow Performance
Expected with CPU offloading. To improve:
1. Use smaller model variant
2. Enable quantization
3. Reduce max context length