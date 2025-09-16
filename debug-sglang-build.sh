#!/bin/bash
# Debug SGLang build issues and verify PyTorch compatibility

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 SGLang Build Debug Tool${NC}"
echo "=========================="
echo

# Function to run debug commands in container
debug_in_container() {
    local container_name="$1"
    local command="$2"
    echo -e "${BLUE}Running in $container_name: $command${NC}"
    docker exec -it "$container_name" bash -c "$command" || echo -e "${RED}Command failed${NC}"
    echo
}

# Check if source image exists
if docker images sglang:rtx5090-source | grep -q "rtx5090-source"; then
    echo -e "${GREEN}✅ Source image found${NC}"
    IMAGE_EXISTS=true
else
    echo -e "${YELLOW}⚠️  Source image not found${NC}"
    IMAGE_EXISTS=false
fi

# Check if container is running
if docker ps | grep -q "qwen3-32b-awq-source"; then
    echo -e "${GREEN}✅ Container is running${NC}"
    CONTAINER_RUNNING=true
else
    echo -e "${YELLOW}⚠️  Container not running${NC}"
    CONTAINER_RUNNING=false
fi

echo

# Test 1: PyTorch Installation Check
echo -e "${BLUE}1. PyTorch Installation Check${NC}"
if [ "$IMAGE_EXISTS" = true ]; then
    docker run --rm --gpus all sglang:rtx5090-source python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')
"
else
    echo "Skipping - image not built"
fi
echo

# Test 2: SGLang Import Test
echo -e "${BLUE}2. SGLang Import Test${NC}"
if [ "$IMAGE_EXISTS" = true ]; then
    docker run --rm --gpus all sglang:rtx5090-source python -c "
try:
    import sglang
    print('✅ SGLang imported successfully')
    print(f'SGLang version: {getattr(sglang, \"__version__\", \"unknown\")}')
except ImportError as e:
    print(f'❌ SGLang import failed: {e}')
    exit(1)
"
else
    echo "Skipping - image not built"
fi
echo

# Test 3: sgl_kernel Import Test
echo -e "${BLUE}3. sgl_kernel Import Test${NC}"
if [ "$IMAGE_EXISTS" = true ]; then
    docker run --rm --gpus all sglang:rtx5090-source python -c "
try:
    from sglang.srt.layers import sgl_kernel
    print('✅ sgl_kernel imported successfully')
    print('This confirms the source build resolved the PyTorch 2.7.0 compatibility issue')
except ImportError as e:
    print(f'❌ sgl_kernel import failed: {e}')
    print('This indicates the source build needs investigation')

    # Try to get more details
    try:
        import torch
        import sglang
        print(f'PyTorch: {torch.__version__}')
        print(f'SGLang imported, but sgl_kernel failed')
    except Exception as e2:
        print(f'Additional error: {e2}')
    exit(1)
"
else
    echo "Skipping - image not built"
fi
echo

# Test 4: CUDA Kernel Compilation Check
echo -e "${BLUE}4. CUDA Kernel Compilation Check${NC}"
if [ "$IMAGE_EXISTS" = true ]; then
    echo "Checking for compiled CUDA kernels..."
    docker run --rm --gpus all sglang:rtx5090-source find /usr/local/lib/python3.11/site-packages -name "*.so" | grep -E "(sgl|kernel)" | head -10
    echo
    docker run --rm --gpus all sglang:rtx5090-source python -c "
import torch
from torch.utils.cpp_extension import CUDA_HOME
print(f'CUDA_HOME: {CUDA_HOME}')
print(f'CUDA available for compilation: {torch.cuda.is_available()}')

# Check torch extensions
try:
    import torch.utils.cpp_extension
    print('✅ Torch C++ extensions available')
except ImportError:
    print('❌ Torch C++ extensions not available')
"
else
    echo "Skipping - image not built"
fi
echo

# Test 5: Runtime Container Debug (if running)
echo -e "${BLUE}5. Runtime Container Debug${NC}"
if [ "$CONTAINER_RUNNING" = true ]; then
    debug_in_container "qwen3-32b-awq-source" "python -c 'import torch; print(f\"GPU memory: {torch.cuda.memory_allocated(0)/1024/1024:.1f}MB\")'"
    debug_in_container "qwen3-32b-awq-source" "python -c 'from sglang.srt.layers import sgl_kernel; print(\"sgl_kernel working in runtime\")'"
    debug_in_container "qwen3-32b-awq-source" "ps aux | grep python"
else
    echo "Container not running - skipping runtime tests"
fi
echo

# Test 6: Build Logs Analysis
echo -e "${BLUE}6. Build Logs Analysis${NC}"
if [ "$IMAGE_EXISTS" = true ]; then
    echo "Checking for potential build warnings..."
    docker run --rm sglang:rtx5090-source python -c "
import sys
import pkg_resources

# Check for conflicting packages
packages = [pkg.key for pkg in pkg_resources.working_set]
torch_packages = [p for p in packages if 'torch' in p]
print(f'Torch-related packages: {torch_packages}')

# Check CUDA compilation flags
import torch.utils.cpp_extension
print(f'C++ compiler: {torch.utils.cpp_extension.CppExtension}')
"
else
    echo "Image not built - check build logs when building"
fi
echo

# Test 7: Symbol Compatibility Test
echo -e "${BLUE}7. Symbol Compatibility Test${NC}"
if [ "$IMAGE_EXISTS" = true ]; then
    echo "Testing for the specific symbol error that was causing issues..."
    docker run --rm --gpus all sglang:rtx5090-source python -c "
try:
    # This is the specific operation that was failing
    import torch
    # Test CUDA device setting (the failing symbol was related to this)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print('✅ CUDA device setting works - symbol issue resolved')
    else:
        print('⚠️  CUDA not available in container')

    from sglang.srt.layers import sgl_kernel
    print('✅ sgl_kernel import successful - build fixed the compatibility issue')

except Exception as e:
    print(f'❌ Symbol compatibility test failed: {e}')
    print('The undefined symbol issue may still exist')
"
else
    echo "Skipping - image not built"
fi

echo
echo -e "${BLUE}🎯 Debug Summary${NC}"
echo "================"

if [ "$IMAGE_EXISTS" = true ]; then
    echo -e "${GREEN}✅ Source image built successfully${NC}"
    if [ "$CONTAINER_RUNNING" = true ]; then
        echo -e "${GREEN}✅ Container is running${NC}"
        echo
        echo -e "${BLUE}📋 Next Steps:${NC}"
        echo "1. Run full API test: ./test-sglang-api.sh"
        echo "2. Monitor performance: nvidia-smi -l 1"
        echo "3. Check logs: docker logs -f qwen3-32b-awq-source"
    else
        echo -e "${YELLOW}⚠️  Container not running${NC}"
        echo
        echo -e "${BLUE}📋 Next Steps:${NC}"
        echo "1. Deploy container: ./deploy-sglang-source-rtx5090.sh"
        echo "2. Test API: ./test-sglang-api.sh"
    fi
else
    echo -e "${RED}❌ Source image not built${NC}"
    echo
    echo -e "${BLUE}📋 Next Steps:${NC}"
    echo "1. Build the image: ./build-sglang-rtx5090.sh"
    echo "2. Monitor build process for errors"
    echo "3. Check CUDA toolkit installation"
fi

echo
echo -e "${BLUE}🔧 Troubleshooting Commands:${NC}"
echo "Build image:      ./build-sglang-rtx5090.sh"
echo "Deploy container: ./deploy-sglang-source-rtx5090.sh"
echo "Test API:         ./test-sglang-api.sh"
echo "View logs:        docker logs qwen3-32b-awq-source"
echo "Debug build:      docker run --rm -it sglang:rtx5090-source bash"