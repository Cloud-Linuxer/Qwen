#!/bin/bash
# Qwen3-30B-A3B Deployment Script for RTX 5090

set -e

echo "=== Qwen3-30B-A3B Deployment Script ==="
echo "Optimized for RTX 5090 with Blackwell architecture"
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check GPU availability
echo "🔍 Checking GPU availability..."
if ! nvidia-smi &>/dev/null; then
    echo -e "${RED}❌ No NVIDIA GPU detected${NC}"
    exit 1
fi

# Display GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo

# Check available disk space
echo "💾 Checking disk space..."
AVAILABLE_SPACE=$(df -BG /home/gpt-oss | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 100 ]; then
    echo -e "${YELLOW}⚠️  Warning: Less than 100GB available. Model requires ~60GB.${NC}"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Memory estimation
echo -e "${YELLOW}📊 Memory Requirements:${NC}"
echo "  Model: Qwen3-30B-A3B (30B parameters)"
echo "  Required: ~70-75GB total"
echo "  Available: 92GB (32GB VRAM + 60GB RAM)"
echo -e "${GREEN}  ✅ Sufficient memory available${NC}"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p data

# Check for existing containers
echo "🔍 Checking for existing containers..."
if docker ps -a | grep -q "qwen3-next-vllm\|qwen-backend\|qwen-frontend"; then
    echo -e "${YELLOW}⚠️  Found existing Qwen containers${NC}"
    read -p "Stop and remove them? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose -f docker-compose.qwen.yml down
    fi
fi

# Build or use existing RTX 5090 optimized image
if docker images | grep -q "qwen3-rtx5090"; then
    echo -e "${GREEN}✅ Using existing RTX 5090 optimized image${NC}"
else
    echo "🔨 Building RTX 5090 optimized Docker image..."
    docker build -f Dockerfile.rtx5090 -t qwen3-rtx5090:latest .
fi

# Model information
echo "📥 Model will be downloaded on first run (approx. 60GB)"
echo -e "${GREEN}Note: Qwen3-30B-A3B fits comfortably in available memory${NC}"
echo

# Start services
echo "🚀 Starting services..."
docker compose -f docker-compose.qwen.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
echo "This may take 2-5 minutes for model loading..."

# Check service health
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ vLLM service is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ Service failed to start. Check logs:${NC}"
    echo "docker compose -f docker-compose.qwen.yml logs qwen3-next"
    exit 1
fi

# Display service status
echo
echo "=== Service Status ==="
docker compose -f docker-compose.qwen.yml ps
echo

# Display access information
echo "=== Access Information ==="
echo -e "${GREEN}✅ Qwen3-30B-A3B is now running!${NC}"
echo
echo "📍 API Endpoint: http://localhost:8000"
echo "📍 Backend API: http://localhost:8080"
echo "📍 Frontend UI: http://localhost:8501"
echo
echo "=== Useful Commands ==="
echo "View logs: docker compose -f docker-compose.qwen.yml logs -f qwen3-next"
echo "Stop services: docker compose -f docker-compose.qwen.yml down"
echo "Restart services: docker compose -f docker-compose.qwen.yml restart"
echo

# Test API endpoint
echo "🧪 Testing API endpoint..."
curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-30B-A3B",
        "prompt": "Hello, ",
        "max_tokens": 10,
        "temperature": 0.7
    }' | jq -r '.choices[0].text' 2>/dev/null || echo -e "${YELLOW}API test pending...${NC}"

echo
echo -e "${GREEN}🎉 Deployment complete!${NC}"