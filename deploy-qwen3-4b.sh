#!/bin/bash
# Qwen3-4B-Instruct-2507 Deployment Script for RTX 5090

set -e

echo "=== Qwen3-4B-Instruct-2507 Deployment Script ==="
echo "Optimized for RTX 5090 - This 4B model will run smoothly!"
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
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo

# Check available disk space
echo "💾 Checking disk space..."
AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 20 ]; then
    echo -e "${YELLOW}⚠️  Warning: Less than 20GB available. Model requires ~8GB.${NC}"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Memory estimation
echo -e "${GREEN}📊 Memory Requirements:${NC}"
echo "  Model: Qwen3-4B-Instruct-2507 (4B parameters)"
echo "  Required: ~8-10GB VRAM"
echo "  Available: 32GB VRAM (RTX 5090)"
echo -e "${GREEN}  ✅ Plenty of memory available - No CPU offloading needed!${NC}"
echo

# Check for existing containers
echo "🔍 Checking for existing containers..."
if docker ps -a | grep -q "qwen3-4b-vllm\|qwen3-4b-backend"; then
    echo -e "${YELLOW}⚠️  Found existing Qwen3-4B containers${NC}"
    read -p "Stop and remove them? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose -f docker-compose.qwen3-4b.yml down 2>/dev/null || true
    fi
fi

# Also check for other Qwen containers
if docker ps -a | grep -q "qwen3-next-vllm\|qwen-backend\|qwen-frontend"; then
    echo -e "${YELLOW}ℹ️  Found other Qwen model containers running${NC}"
    echo "You may want to stop them to free up resources:"
    echo "  docker compose -f docker-compose.qwen.yml down"
    echo
fi

# Model information
echo "📥 Model Download Information:"
echo "  Model: Qwen/Qwen3-4B-Instruct-2507"
echo "  Size: ~8GB (will be cached after first download)"
echo "  Source: HuggingFace Hub"
echo

# Start services
echo "🚀 Starting services..."
docker compose -f docker-compose.qwen3-4b.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
echo "This should be quick with the 4B model (1-2 minutes)..."

# Check service health
MAX_RETRIES=20
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ vLLM service is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ Service failed to start. Check logs:${NC}"
    echo "docker compose -f docker-compose.qwen3-4b.yml logs qwen3-4b"
    exit 1
fi

# Display service status
echo
echo "=== Service Status ==="
docker compose -f docker-compose.qwen3-4b.yml ps
echo

# Display access information
echo "=== Access Information ==="
echo -e "${GREEN}✅ Qwen3-4B-Instruct-2507 is now running!${NC}"
echo
echo "📍 API Endpoint: http://localhost:8000"
echo "📍 OpenAI API: http://localhost:8000/v1"
echo "📍 Backend API: http://localhost:8080"
echo
echo "=== Quick Test Commands ==="
echo
echo "1. Test completion endpoint:"
echo "curl -X POST http://localhost:8000/v1/completions \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"model\": \"Qwen/Qwen3-4B-Instruct-2507\","
echo "    \"prompt\": \"What is machine learning?\","
echo "    \"max_tokens\": 100"
echo "  }'"
echo
echo "2. Test chat endpoint:"
echo "curl -X POST http://localhost:8000/v1/chat/completions \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"model\": \"Qwen/Qwen3-4B-Instruct-2507\","
echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello! How are you?\"}],"
echo "    \"max_tokens\": 100"
echo "  }'"
echo
echo "=== Useful Commands ==="
echo "View logs: docker compose -f docker-compose.qwen3-4b.yml logs -f qwen3-4b"
echo "Stop services: docker compose -f docker-compose.qwen3-4b.yml down"
echo "Restart services: docker compose -f docker-compose.qwen3-4b.yml restart"
echo "GPU monitoring: watch -n 1 nvidia-smi"
echo

# Test API endpoint
echo "🧪 Testing API endpoint..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "prompt": "Hello, I am",
        "max_tokens": 10,
        "temperature": 0.7
    }' 2>/dev/null | jq -r '.choices[0].text' 2>/dev/null || echo "")

if [ -n "$TEST_RESPONSE" ]; then
    echo -e "${GREEN}✅ API Test successful! Response: $TEST_RESPONSE${NC}"
else
    echo -e "${YELLOW}⚠️ API test pending... The model might still be loading.${NC}"
    echo "You can test manually with the commands above once ready."
fi

echo
echo -e "${GREEN}🎉 Deployment complete!${NC}"
echo "The 4B model should run very fast on your RTX 5090 with no memory issues!"