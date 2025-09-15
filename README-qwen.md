# Qwen3-Next-80B-A3B-Instruct Deployment

## Overview
This deployment runs Qwen3-Next-80B-A3B-Instruct model using vLLM with Docker Compose.

## System Requirements
- GPU: NVIDIA RTX 5090 (32GB VRAM)
- RAM: 60GB+ (for CPU offloading)
- Disk: 200GB+ free space
- Docker & Docker Compose
- NVIDIA Container Toolkit

## Model Specifications
- **Parameters**: 80B total, 3B activated (MoE)
- **Context Length**: Up to 262K tokens
- **Quantization**: GPTQ 4-bit (optional)
- **CPU Offloading**: 100GB to handle memory constraints

## Quick Start

```bash
# Deploy the model
./deploy-qwen.sh

# Or manually:
docker-compose -f docker-compose.qwen.yml up -d
```

## Access Points
- **vLLM API**: http://localhost:8000
- **Backend API**: http://localhost:8080
- **Frontend UI**: http://localhost:8501

## API Usage

### OpenAI-compatible API
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 1000,
    "temperature": 0.7
  }'
```

### Python Client
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require auth
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-Next-80B-A3B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=1000,
    temperature=0.7
)
```

## Configuration

### Memory Optimization
The deployment uses:
- **GPU Memory**: 95% utilization (~30GB)
- **CPU Offloading**: 100GB for model weights
- **Reduced Context**: 16K tokens (adjustable)

### RTX 5090 Compatibility
Due to RTX 5090's sm_120 architecture not being fully supported by PyTorch:
- Using compatibility workarounds via environment variables
- CPU offloading enabled for stability
- Custom all-reduce disabled

## Monitoring

```bash
# View logs
docker-compose -f docker-compose.qwen.yml logs -f qwen3-next

# Check GPU usage
nvidia-smi -l 1

# Service status
docker-compose -f docker-compose.qwen.yml ps
```

## Troubleshooting

### Out of Memory
- Reduce `--max-model-len` in docker-compose.qwen.yml
- Increase `--cpu-offload-gb` value
- Consider using quantization

### Slow Performance
- Expected due to CPU offloading
- Consider smaller models or cloud deployment for production

### RTX 5090 Issues
- If CUDA errors occur, check PyTorch compatibility
- Monitor for official RTX 5090 support updates

## Cleanup

```bash
# Stop services
docker-compose -f docker-compose.qwen.yml down

# Remove volumes and images
docker-compose -f docker-compose.qwen.yml down -v --rmi all
```

## Notes
- Initial model download takes 10-30 minutes (~160GB)
- Performance is limited by CPU offloading requirements
- For production, consider cloud GPUs with more VRAM