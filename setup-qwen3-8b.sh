#!/bin/bash

# Setup script for Qwen3-8B model with SGLang
set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Setting up Qwen3-8B with SGLang${NC}"
echo "======================================================="
echo "- Model: Qwen/Qwen3-8B from Hugging Face"
echo "- Framework: SGLang"
echo "- Optimized for inference"
echo "======================================================="
echo

# Check prerequisites
echo -e "${BLUE}🔍 Checking Prerequisites${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found. Installing...${NC}"
    apt-get update && apt-get install -y python3 python3-pip
fi

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: NVIDIA driver not detected${NC}"
    echo "GPU acceleration may not be available"
fi

# Create virtual environment
echo -e "${BLUE}📦 Setting up Python environment${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install required packages
echo -e "${BLUE}📥 Installing dependencies${NC}"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate sentencepiece protobuf
pip install sglang

# Download Qwen3-8B model
echo -e "${BLUE}📥 Downloading Qwen3-8B model from Hugging Face${NC}"
python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_name = "Qwen/Qwen3-8B"
cache_dir = os.path.expanduser("~/.cache/huggingface")

print(f"Downloading model: {model_name}")
print(f"Cache directory: {cache_dir}")

try:
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Download model
    print("Downloading model weights (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("✅ Model downloaded successfully!")
    print(f"Model location: {cache_dir}")

except Exception as e:
    print(f"❌ Error downloading model: {e}")
    exit(1)
EOF

# Create SGLang deployment script
echo -e "${BLUE}📝 Creating SGLang deployment script${NC}"
cat > deploy-qwen3-8b-sglang.sh << 'SCRIPT'
#!/bin/bash

# Deploy Qwen3-8B with SGLang

echo "🚀 Deploying Qwen3-8B with SGLang..."

# Kill any existing SGLang processes
pkill -f sglang.launch_server || true

# Start SGLang server
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --tp 1 \
    --trust-remote-code \
    --quantization none \
    --max-total-tokens 8192 \
    --max-prefill-tokens 4096 \
    --chunked-prefill-size 2048 \
    --mem-fraction-static 0.85 \
    --disable-cuda-graph \
    --attention-backend flashinfer &

echo "✅ SGLang server starting on port 8000..."
echo "Wait for model to load, then access at http://localhost:8000"
SCRIPT

chmod +x deploy-qwen3-8b-sglang.sh

# Create test script
echo -e "${BLUE}📝 Creating test script${NC}"
cat > test-qwen3-8b.py << 'EOF'
#!/usr/bin/env python3

import requests
import json
import time

def test_model():
    """Test Qwen3-8B model through SGLang API"""

    url = "http://localhost:8000/v1/chat/completions"

    # Test prompt
    payload = {
        "model": "Qwen/Qwen3-8B",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": "What is the capital of France? Please provide a brief answer."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    try:
        print("🔍 Testing Qwen3-8B model...")
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            print("✅ Test successful!")
            print(f"Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"❌ Test failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to SGLang server. Make sure it's running.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Wait a bit for server to be ready
    print("Waiting 5 seconds for server to initialize...")
    time.sleep(5)
    test_model()
EOF

chmod +x test-qwen3-8b.py

# Create Docker deployment option
echo -e "${BLUE}📝 Creating Docker deployment script${NC}"
cat > deploy-qwen3-8b-docker.sh << 'EOF'
#!/bin/bash

# Docker deployment for Qwen3-8B with SGLang

echo "🚀 Deploying Qwen3-8B with SGLang in Docker..."

# Stop and remove existing container
docker stop qwen3-8b-sglang 2>/dev/null
docker rm qwen3-8b-sglang 2>/dev/null

# Run SGLang container with Qwen3-8B
docker run -d \
  --name qwen3-8b-sglang \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 16g \
  lmsysorg/sglang:latest \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --tp 1 \
    --trust-remote-code \
    --quantization none \
    --max-total-tokens 8192 \
    --mem-fraction-static 0.85

echo "✅ Container started. Check logs with: docker logs -f qwen3-8b-sglang"
EOF

chmod +x deploy-qwen3-8b-docker.sh

echo -e "${GREEN}✅ Setup complete!${NC}"
echo
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. Deploy with native Python: ./deploy-qwen3-8b-sglang.sh"
echo "2. Or deploy with Docker: ./deploy-qwen3-8b-docker.sh"
echo "3. Test the model: ./test-qwen3-8b.py"
echo
echo -e "${BLUE}📊 Model Information:${NC}"
echo "Model: Qwen/Qwen3-8B"
echo "Parameters: 8B"
echo "Cache location: ~/.cache/huggingface"
echo "API endpoint: http://localhost:8000"
echo
echo -e "${YELLOW}⚠️  Note:${NC}"
echo "The model requires approximately 16GB of GPU memory for inference."
echo "For quantized versions, consider using Qwen3-8B-AWQ or Qwen3-8B-GPTQ."