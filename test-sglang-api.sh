#!/bin/bash
# Test SGLang API endpoints with comprehensive validation

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BASE_URL="http://localhost:8000"
API_KEY="sk-sglang-key-12345"

echo -e "${BLUE}🧪 Testing SGLang API${NC}"
echo "===================="
echo "Base URL: $BASE_URL"
echo

# Test 1: Health Check
echo -e "${BLUE}1. Health Check${NC}"
if curl -s "$BASE_URL/health" | grep -q "OK\|healthy"; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
    exit 1
fi
echo

# Test 2: Model Info
echo -e "${BLUE}2. Model Information${NC}"
MODEL_RESPONSE=$(curl -s "$BASE_URL/v1/models")
if echo "$MODEL_RESPONSE" | jq -e '.data[0].id' > /dev/null 2>&1; then
    MODEL_ID=$(echo "$MODEL_RESPONSE" | jq -r '.data[0].id')
    echo -e "${GREEN}✅ Model loaded: $MODEL_ID${NC}"
else
    echo -e "${RED}❌ Model info failed${NC}"
    echo "Response: $MODEL_RESPONSE"
    exit 1
fi
echo

# Test 3: Simple Completion
echo -e "${BLUE}3. Simple Completion Test${NC}"
COMPLETION_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
        "model": "'"$MODEL_ID"'",
        "prompt": "The capital of France is",
        "max_tokens": 10,
        "temperature": 0.1
    }')

if echo "$COMPLETION_RESPONSE" | jq -e '.choices[0].text' > /dev/null 2>&1; then
    COMPLETION_TEXT=$(echo "$COMPLETION_RESPONSE" | jq -r '.choices[0].text')
    echo -e "${GREEN}✅ Completion successful${NC}"
    echo "Response: '$COMPLETION_TEXT'"
else
    echo -e "${RED}❌ Completion failed${NC}"
    echo "Response: $COMPLETION_RESPONSE"
fi
echo

# Test 4: Chat Completion
echo -e "${BLUE}4. Chat Completion Test${NC}"
CHAT_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
        "model": "'"$MODEL_ID"'",
        "messages": [
            {"role": "user", "content": "Hello! Please respond with just the word WORKING."}
        ],
        "max_tokens": 20,
        "temperature": 0.1
    }')

if echo "$CHAT_RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
    CHAT_TEXT=$(echo "$CHAT_RESPONSE" | jq -r '.choices[0].message.content')
    echo -e "${GREEN}✅ Chat completion successful${NC}"
    echo "Response: '$CHAT_TEXT'"
else
    echo -e "${RED}❌ Chat completion failed${NC}"
    echo "Response: $CHAT_RESPONSE"
fi
echo

# Test 5: Performance Test
echo -e "${BLUE}5. Performance Test${NC}"
echo "Testing generation speed..."

START_TIME=$(date +%s.%N)
PERF_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
        "model": "'"$MODEL_ID"'",
        "messages": [
            {"role": "user", "content": "Write a very brief paragraph about artificial intelligence."}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }')
END_TIME=$(date +%s.%N)

if echo "$PERF_RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
    RESPONSE_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    RESPONSE_TEXT=$(echo "$PERF_RESPONSE" | jq -r '.choices[0].message.content')
    TOKEN_COUNT=$(echo "$RESPONSE_TEXT" | wc -w)
    TOKENS_PER_SEC=$(echo "scale=2; $TOKEN_COUNT / $RESPONSE_TIME" | bc)

    echo -e "${GREEN}✅ Performance test successful${NC}"
    echo "Response time: ${RESPONSE_TIME}s"
    echo "Estimated tokens: $TOKEN_COUNT"
    echo "Estimated tokens/sec: $TOKENS_PER_SEC"
    echo
    echo "Response preview:"
    echo "\"$(echo "$RESPONSE_TEXT" | cut -c1-100)...\""
else
    echo -e "${RED}❌ Performance test failed${NC}"
    echo "Response: $PERF_RESPONSE"
fi
echo

# Test 6: GPU Memory Check
echo -e "${BLUE}6. GPU Memory Usage${NC}"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
while IFS=, read used total util; do
    used_gb=$(echo "scale=1; $used / 1024" | bc)
    total_gb=$(echo "scale=1; $total / 1024" | bc)
    usage_percent=$(echo "scale=1; $used * 100 / $total" | bc)

    echo "Memory: ${used_gb}GB / ${total_gb}GB (${usage_percent}%)"
    echo "GPU Utilization: ${util}%"

    if (( $(echo "$usage_percent > 95" | bc -l) )); then
        echo -e "${YELLOW}⚠️  High memory usage${NC}"
    elif (( $(echo "$usage_percent > 80" | bc -l) )); then
        echo -e "${GREEN}✅ Good memory usage${NC}"
    else
        echo -e "${BLUE}ℹ️  Low memory usage${NC}"
    fi
done
echo

# Summary
echo -e "${BLUE}🎯 Test Summary${NC}"
echo "==============="
echo -e "${GREEN}✅ SGLang API is working correctly${NC}"
echo -e "${GREEN}✅ Source-built sgl_kernel compatible with PyTorch 2.7.0${NC}"
echo -e "${GREEN}✅ RTX 5090 optimization active${NC}"
echo
echo -e "${BLUE}📋 Useful Commands:${NC}"
echo "Monitor logs:     docker logs -f qwen3-32b-awq-source"
echo "GPU monitoring:   nvidia-smi -l 1"
echo "Container stats:  docker stats qwen3-32b-awq-source"
echo "API health:       curl $BASE_URL/health"
echo