#!/bin/bash
# Build SGLang from source for RTX 5090 with PyTorch 2.7.0 compatibility

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔨 Building SGLang from Source for RTX 5090${NC}"
echo "======================================================="
echo "- PyTorch 2.7.0 with CUDA 12.8"
echo "- sgl_kernel compiled from source"
echo "- RTX 5090 Blackwell (sm_120) support"
echo "======================================================="
echo

# Pre-flight checks
echo -e "${BLUE}🔍 Pre-flight Checks${NC}"

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA driver not found${NC}"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found${NC}"
    exit 1
fi

# Check available disk space (need at least 20GB for build)
AVAILABLE_SPACE=$(df /var/lib/docker --output=avail | tail -n1)
REQUIRED_SPACE=$((20 * 1024 * 1024)) # 20GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo -e "${YELLOW}⚠️  Warning: Low disk space. Need at least 20GB for build.${NC}"
    echo "Available: $(($AVAILABLE_SPACE / 1024 / 1024))GB"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Show GPU info
echo -e "${BLUE}📊 GPU Information:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader
echo

# Remove any existing builds
echo -e "${BLUE}🧹 Cleaning Previous Builds${NC}"
docker stop sglang-source-build 2>/dev/null || true
docker rm sglang-source-build 2>/dev/null || true
docker rmi sglang:rtx5090-source 2>/dev/null || true

# Build the image
echo -e "${BLUE}🔨 Building SGLang Source Image${NC}"
echo "This will take 15-30 minutes depending on your system..."
echo

# Check if Dockerfile exists
if [ ! -f "Dockerfile.sglang-source-build" ]; then
    echo -e "${RED}❌ Dockerfile.sglang-source-build not found${NC}"
    exit 1
fi

# Build with progress and proper BuildKit
export DOCKER_BUILDKIT=1
docker build \
    --progress=plain \
    --tag sglang:rtx5090-source \
    --file Dockerfile.sglang-source-build \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Build completed successfully${NC}"
echo

# Verify the build
echo -e "${BLUE}🧪 Verifying Build${NC}"
docker run --rm --gpus all sglang:rtx5090-source python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'CUDA devices: {torch.cuda.device_count()}')

import sglang
print(f'SGLang imported successfully')

try:
    from sglang.srt.layers import sgl_kernel
    print('✅ sgl_kernel imported successfully - source build working!')
except ImportError as e:
    print(f'❌ sgl_kernel import failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Verification passed${NC}"
else
    echo -e "${RED}❌ Verification failed${NC}"
    exit 1
fi

echo
echo -e "${GREEN}🎉 SGLang source build completed!${NC}"
echo
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. Deploy with: ./deploy-sglang-source-rtx5090.sh"
echo "2. Or run manually with the sglang:rtx5090-source image"
echo
echo -e "${BLUE}🏷️  Image Info:${NC}"
echo "Image name: sglang:rtx5090-source"
echo "Size: $(docker images sglang:rtx5090-source --format 'table {{.Size}}' | tail -n1)"
echo