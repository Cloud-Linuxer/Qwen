#!/bin/bash
# SGLang Deployment Script for RTX 5090 - No CPU Offloading
# Optimized for maximum GPU performance

set -e

echo "=== SGLang Deployment Script for Qwen Models ==="
echo "RTX 5090 Optimized - Full GPU Memory Utilization"
echo "================================================"
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Model selection
echo -e "${BLUE}📦 Available Models:${NC}"
echo "1) Qwen3-4B-Instruct-2507 (Recommended for speed)"
echo "2) Qwen3-30B-A3B (Balanced performance)"
echo "3) Qwen2.5-32B-Instruct (Latest version)"
echo "4) Qwen2.5-7B-Instruct (Lightweight)"
echo
read -p "Select model (1-4): " MODEL_CHOICE

case $MODEL_CHOICE in
    1)
        MODEL="Qwen/Qwen3-4B-Instruct-2507"
        MODEL_NAME="qwen3-4b"
        MAX_TOKENS=16384
        ;;
    2)
        MODEL="Qwen/Qwen3-30B-A3B"
        MODEL_NAME="qwen3-30b"
        MAX_TOKENS=8192
        ;;
    3)
        MODEL="Qwen/Qwen2.5-32B-Instruct"
        MODEL_NAME="qwen2.5-32b"
        MAX_TOKENS=8192
        ;;
    4)
        MODEL="Qwen/Qwen2.5-7B-Instruct"
        MODEL_NAME="qwen2.5-7b"
        MAX_TOKENS=16384
        ;;
    *)
        echo -e "${RED}❌ Invalid selection${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}✅ Selected: $MODEL${NC}"
echo

# Check GPU availability
echo "🔍 Checking GPU availability..."
if ! nvidia-smi &>/dev/null; then
    echo -e "${RED}❌ No NVIDIA GPU detected${NC}"
    exit 1
fi

# Display GPU info
echo -e "${BLUE}📊 GPU Information:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader
echo

# Check available VRAM
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo -e "${BLUE}💾 Available VRAM: ${VRAM_FREE}MB${NC}"

# Memory requirements check
case $MODEL_CHOICE in
    1)
        REQUIRED_VRAM=8000
        ;;
    2)
        REQUIRED_VRAM=25000
        ;;
    3)
        REQUIRED_VRAM=26000
        ;;
    4)
        REQUIRED_VRAM=14000
        ;;
esac

if [ "$VRAM_FREE" -lt "$REQUIRED_VRAM" ]; then
    echo -e "${RED}❌ Insufficient VRAM. Required: ${REQUIRED_VRAM}MB, Available: ${VRAM_FREE}MB${NC}"
    echo -e "${YELLOW}💡 Tip: Try a smaller model or stop other GPU processes${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Sufficient VRAM available${NC}"
echo

# Check for existing containers
echo "🔍 Checking for existing containers..."
EXISTING_CONTAINERS=$(docker ps -a --format "{{.Names}}" | grep -E "sglang|qwen|vllm" || true)
if [ ! -z "$EXISTING_CONTAINERS" ]; then
    echo -e "${YELLOW}⚠️  Found existing containers:${NC}"
    echo "$EXISTING_CONTAINERS"
    read -p "Stop and remove them? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$EXISTING_CONTAINERS" | xargs -r docker stop
        echo "$EXISTING_CONTAINERS" | xargs -r docker rm
    fi
fi

# Create updated docker-compose file
echo "📝 Creating optimized SGLang configuration..."
cat > docker-compose.sglang-optimized.yml << EOF
version: '3.8'

services:
  sglang-${MODEL_NAME}:
    image: lmsysorg/sglang:latest
    container_name: sglang-${MODEL_NAME}
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "127.0.0.1:8000:8000"  # Localhost only for security
      - "127.0.0.1:8001:8001"  # Metrics port
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./models:/models
      - /tmp:/tmp
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HF_TOKEN=\${HF_TOKEN:-}
      - HUGGING_FACE_HUB_TOKEN=\${HF_TOKEN:-}
      - HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-0}
      # RTX 5090 Blackwell optimization
      - TORCH_CUDA_ARCH_LIST=7.0;7.5;8.0;8.6;8.9;9.0;12.0+PTX
      - CUDA_LAUNCH_BLOCKING=0
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      # SGLang specific
      - SGLANG_PROFILING=1
      - SGLANG_LOG_LEVEL=INFO
    command: >
      python -m sglang.launch_server
      --model-path ${MODEL}
      --host 0.0.0.0
      --port 8000
      --dtype auto
      --max-total-tokens ${MAX_TOKENS}
      --mem-fraction-static 0.95
      --trust-remote-code
      --enable-flashinfer
      --enable-torch-compile
      --cuda-graph-max-bs 16
      --schedule-policy lpm
      --lru-cache-size 1024
      --prefill-target-length 4096
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 180s
    restart: unless-stopped
    shm_size: '32gb'
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 65536
        hard: 65536

networks:
  default:
    name: sglang-network
    driver: bridge
EOF

echo -e "${GREEN}✅ Configuration created${NC}"
echo

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << 'EOF'
# Hugging Face Token (optional, for gated models)
HF_TOKEN=

# Offline mode (set to 1 to use cached models only)
HF_HUB_OFFLINE=0
EOF
    echo -e "${YELLOW}ℹ️  Created .env file. Add your HF_TOKEN if needed for gated models${NC}"
fi

# Start services
echo -e "${BLUE}🚀 Starting SGLang server...${NC}"
docker compose -f docker-compose.sglang-optimized.yml up -d

# Wait for service to be ready
echo "⏳ Waiting for SGLang to initialize..."
echo "This may take 1-3 minutes depending on model size..."

MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ SGLang server is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ Service failed to start. Checking logs...${NC}"
    docker compose -f docker-compose.sglang-optimized.yml logs --tail=50
    exit 1
fi

echo
echo "=== Service Status ==="
docker compose -f docker-compose.sglang-optimized.yml ps
echo

# Display API information
echo -e "${BLUE}=== API Endpoints ===${NC}"
echo "📍 Main API: http://localhost:8000"
echo "📍 Health: http://localhost:8000/health"
echo "📍 Models: http://localhost:8000/v1/models"
echo "📍 Metrics: http://localhost:8001/metrics"
echo

# Test the API
echo -e "${BLUE}🧪 Testing API...${NC}"
echo

# Test completion
echo "Testing completion endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"prompt\": \"What is machine learning in one sentence?\",
        \"max_tokens\": 50,
        \"temperature\": 0.7
    }" 2>/dev/null)

if echo "$RESPONSE" | jq -e '.choices[0].text' > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Completion test successful!${NC}"
    echo "Response: $(echo "$RESPONSE" | jq -r '.choices[0].text')"
else
    echo -e "${YELLOW}⚠️ Completion test pending...${NC}"
fi

echo

# Display useful commands
echo -e "${BLUE}=== Useful Commands ===${NC}"
echo "📊 Monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo
echo "📝 View logs:"
echo "  docker compose -f docker-compose.sglang-optimized.yml logs -f"
echo
echo "🔄 Restart service:"
echo "  docker compose -f docker-compose.sglang-optimized.yml restart"
echo
echo "🛑 Stop service:"
echo "  docker compose -f docker-compose.sglang-optimized.yml down"
echo
echo "📈 View metrics:"
echo "  curl http://localhost:8001/metrics"
echo

# Performance tips
echo -e "${BLUE}=== Performance Tips ===${NC}"
echo "• SGLang uses advanced batching for better throughput"
echo "• Flash attention is enabled for faster inference"
echo "• CUDA graphs optimize kernel launches"
echo "• LPM scheduling policy for lower latency"
echo "• No CPU offloading = maximum GPU performance"
echo

echo -e "${GREEN}🎉 SGLang deployment complete!${NC}"
echo "Model: $MODEL"
echo "Max tokens: $MAX_TOKENS"
echo "Memory usage: 95% of VRAM (no CPU offloading)"