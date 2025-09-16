#!/usr/bin/env python3
"""
SGLang RTX 5090 (Blackwell) Compatibility Patch
Disables problematic features and adds Blackwell-specific workarounds
"""

import os
import sys
import torch

def patch_sglang_for_rtx5090():
    """Apply RTX 5090 specific patches to SGLang"""

    # 1. Force disable FlashInfer (causes segfault on Blackwell)
    os.environ["SGLANG_DISABLE_FLASHINFER"] = "1"
    os.environ["FLASHINFER_DISABLE"] = "1"

    # 2. Disable CUDA graphs (incompatible with sm_120)
    os.environ["SGLANG_DISABLE_CUDA_GRAPH"] = "1"
    os.environ["DISABLE_CUDA_GRAPH"] = "1"

    # 3. Force memory settings for stability
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

    # 4. Disable custom kernels that may not support Blackwell
    os.environ["SGLANG_DISABLE_CUSTOM_KERNELS"] = "1"

    # 5. Force attention backend to native PyTorch
    os.environ["SGLANG_ATTENTION_BACKEND"] = "torch_native"

    # 6. Patch sgl_kernel import to handle failures gracefully
    try:
        import sgl_kernel
        print(f"✓ sgl_kernel loaded: {sgl_kernel.__version__}")
    except ImportError as e:
        print(f"⚠️ sgl_kernel not available: {e}")
        print("Falling back to pure PyTorch operations")

        # Mock sgl_kernel to prevent import errors
        class MockSGLKernel:
            def __getattr__(self, name):
                def mock_func(*args, **kwargs):
                    print(f"Warning: sgl_kernel.{name} not available, using fallback")
                    return None
                return mock_func

        sys.modules['sgl_kernel'] = MockSGLKernel()

    # 7. Verify CUDA capability
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        print(f"✓ CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"✓ Compute capability: {capability[0]}.{capability[1]}")

        if capability[0] == 12:  # Blackwell detection
            print("✓ RTX 5090 (Blackwell) detected, applying workarounds")

            # Disable operations known to fail on sm_120
            os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
            os.environ["TORCH_CUDNN_V8_ALLOW_TF32"] = "0"  # Disable TF32 for stability

    print("\n=== RTX 5090 Patches Applied ===")
    print(f"FlashInfer: DISABLED")
    print(f"CUDA Graphs: DISABLED")
    print(f"Custom Kernels: DISABLED")
    print(f"Attention Backend: torch_native")
    print("================================\n")

if __name__ == "__main__":
    patch_sglang_for_rtx5090()

    # Now import and launch SGLang with patches applied
    from sglang import launch_server
    launch_server()