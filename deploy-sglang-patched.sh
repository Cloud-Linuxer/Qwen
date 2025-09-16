#!/bin/bash
# SGLang Patched Deployment for RTX 5090
# All optimizations disabled for maximum compatibility

set -e

echo "=== SGLang RTX 5090 Patched Deployment ==="
echo "Mode: Maximum Compatibility (No Optimizations)"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Clean up
echo -e "${YELLOW}🧹 Cleaning up...${NC}"
docker stop sglang-patched 2>/dev/null || true
docker rm sglang-patched 2>/dev/null || true

# Build patched image
echo -e "${YELLOW}🔨 Building patched image...${NC}"
docker build -f Dockerfile.sglang-patched -t sglang:patched . || {
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
}

echo -e "${GREEN}✅ Image built successfully${NC}"

# Run with absolute minimal settings
echo -e "${YELLOW}🚀 Starting SGLang in compatibility mode...${NC}"
docker run -d \
  --name sglang-patched \
  --runtime nvidia \
  --gpus '"device=0"' \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL_PATH="Qwen/Qwen3-32B-AWQ" \
  -e QUANTIZATION="awq_marlin" \
  -e MAX_TOKENS="2048" \
  -e MEM_FRACTION="0.75" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e CUDA_LAUNCH_BLOCKING=1 \
  -e TORCH_SHOW_CPP_STACKTRACES=1 \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --cap-add SYS_PTRACE \
  sglang:patched

echo -e "${YELLOW}⏳ Waiting for initialization...${NC}"
sleep 10

# Show initial logs
echo -e "${YELLOW}📋 Initial logs:${NC}"
docker logs sglang-patched 2>&1 | head -30

# Monitor startup
echo -e "${YELLOW}⏳ Monitoring startup (60 seconds)...${NC}"
for i in {1..6}; do
    sleep 10
    echo -n "."
    if docker logs sglang-patched 2>&1 | grep -q "Uvicorn running on"; then
        echo -e "\n${GREEN}✅ Server started!${NC}"
        break
    fi
    if docker logs sglang-patched 2>&1 | grep -q "ERROR\|CRITICAL\|Segmentation fault"; then
        echo -e "\n${RED}❌ Error detected:${NC}"
        docker logs sglang-patched 2>&1 | tail -20
        break
    fi
done

echo ""
echo "=== Status ==="
docker ps -a | grep sglang-patched || true

echo ""
echo "=== Commands ==="
echo "Logs: docker logs -f sglang-patched"
echo "Shell: docker exec -it sglang-patched bash"
echo "Stop: docker stop sglang-patched"
echo ""

# Simple test
echo -e "${YELLOW}🧪 Testing server...${NC}"
sleep 5
curl -s http://localhost:8000/health 2>/dev/null && echo -e "${GREEN}✅ Health check passed${NC}" || echo -e "${YELLOW}⚠️ Server still starting...${NC}"