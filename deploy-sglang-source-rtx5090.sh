#!/bin/bash
# Deploy Qwen3-32B-AWQ with custom-built SGLang for RTX 5090

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Deploying SGLang Source Build for RTX 5090${NC}"
echo "=============================================="
echo "- Custom sgl_kernel built for PyTorch 2.7.0"
echo "- Qwen3-32B-AWQ model"
echo "- RTX 5090 optimized settings"
echo "=============================================="
echo

# Check if source image exists
if ! docker images sglang:rtx5090-source | grep -q "rtx5090-source"; then
    echo -e "${RED}❌ Source image not found. Please run ./build-sglang-rtx5090.sh first${NC}"
    exit 1
fi

# GPU check
echo -e "${BLUE}📊 RTX 5090 Status:${NC}"
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader | head -n1)
echo "$GPU_INFO"

# Extract memory info for configuration
TOTAL_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f2 | tr -d ' MiB')
FREE_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f3 | tr -d ' MiB')
TEMP=$(echo "$GPU_INFO" | cut -d',' -f4 | tr -d ' C')

echo -e "${BLUE}Memory: ${FREE_MEMORY}MB free / ${TOTAL_MEMORY}MB total${NC}"
echo -e "${BLUE}Temperature: ${TEMP}°C${NC}"
echo

# Safety checks
if [ "$FREE_MEMORY" -lt 20000 ]; then
    echo -e "${YELLOW}⚠️  Warning: Low GPU memory (${FREE_MEMORY}MB). Qwen3-32B-AWQ needs ~20GB+${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if [ "$TEMP" -gt 80 ]; then
    echo -e "${YELLOW}⚠️  Warning: High GPU temperature (${TEMP}°C). Consider cooling before deployment.${NC}"
fi

# Clean up existing deployment
echo -e "${BLUE}🧹 Cleaning Previous Deployment${NC}"
docker stop qwen3-32b-awq-source 2>/dev/null || true
docker rm qwen3-32b-awq-source 2>/dev/null || true

# Deploy with optimized settings for RTX 5090
echo -e "${BLUE}🚀 Starting Qwen3-32B-AWQ with Source-Built SGLang${NC}"

# Calculate optimal memory fraction based on available memory
if [ "$TOTAL_MEMORY" -gt 30000 ]; then
    MEM_FRACTION="0.95"  # RTX 5090 32GB
    MAX_TOKENS="32768"
else
    MEM_FRACTION="0.90"  # Conservative for other cards
    MAX_TOKENS="16384"
fi

echo "Using memory fraction: $MEM_FRACTION"
echo "Max tokens: $MAX_TOKENS"
echo

docker run -d \
    --name qwen3-32b-awq-source \
    --runtime nvidia \
    --gpus all \
    -p 8000:8000 \
    -p 8001:8001 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/models:/models \
    --shm-size 32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUDA_LAUNCH_BLOCKING=0 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
    sglang:rtx5090-source \
    --model-path Qwen/Qwen3-32B-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype half \
    --quantization awq_marlin \
    --max-total-tokens $MAX_TOKENS \
    --mem-fraction-static $MEM_FRACTION \
    --trust-remote-code \
    --chunked-prefill-size 2048 \
    --enable-torch-compile \
    --disable-custom-all-reduce \
    --kv-cache-dtype fp8

echo -e "${BLUE}⏳ Initializing SGLang server...${NC}"
echo "This may take 2-5 minutes for model loading..."
echo

# Wait for container to start
sleep 10

# Monitor startup
echo -e "${BLUE}📋 Startup Logs:${NC}"
timeout 300 docker logs -f qwen3-32b-awq-source &
LOG_PID=$!

# Wait for health check
echo
echo -e "${BLUE}🔍 Waiting for server to be ready...${NC}"
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Server is ready!${NC}"
        kill $LOG_PID 2>/dev/null || true
        break
    fi
    echo -n "."
    sleep 5
done

echo
echo

# Final status check
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}🎉 Deployment Successful!${NC}"
    echo

    # Show container status
    echo -e "${BLUE}📊 Container Status:${NC}"
    docker ps | grep qwen3-32b-awq-source
    echo

    # Show API endpoints
    echo -e "${BLUE}🌐 API Endpoints:${NC}"
    echo "Health Check: http://localhost:8000/health"
    echo "Model Info:   http://localhost:8000/v1/models"
    echo "Chat API:     http://localhost:8000/v1/chat/completions"
    echo "Completions:  http://localhost:8000/v1/completions"
    echo

    # Test the API
    echo -e "${BLUE}🧪 Testing API:${NC}"
    curl -s http://localhost:8000/v1/models | jq -r '.data[0].id' 2>/dev/null && echo "✅ Model loaded successfully" || echo "⚠️  Model info unavailable"

    echo
    echo -e "${BLUE}📋 Management Commands:${NC}"
    echo "View logs:    docker logs -f qwen3-32b-awq-source"
    echo "Stop server:  docker stop qwen3-32b-awq-source"
    echo "Restart:      docker restart qwen3-32b-awq-source"
    echo "GPU usage:    nvidia-smi -l 1"
    echo

    # Quick test
    echo -e "${BLUE}🔬 Quick Test:${NC}"
    echo "curl -X POST http://localhost:8000/v1/chat/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{"
    echo "    \"model\": \"Qwen/Qwen3-32B-AWQ\","
    echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],"
    echo "    \"max_tokens\": 100"
    echo "  }'"

else
    echo -e "${RED}❌ Deployment Failed${NC}"
    echo
    echo -e "${BLUE}📋 Troubleshooting:${NC}"
    echo "1. Check logs: docker logs qwen3-32b-awq-source"
    echo "2. Check GPU memory: nvidia-smi"
    echo "3. Verify image build: docker images sglang:rtx5090-source"
    echo
    exit 1
fi