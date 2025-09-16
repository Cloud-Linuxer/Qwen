#!/usr/bin/env python3
"""
Minimal test for SGLang with Qwen3-32B-AWQ
"""
import os
import sys

# Disable all problematic features
os.environ["SGLANG_DISABLE_FLASHINFER"] = "1"
os.environ["SGLANG_DISABLE_CUDA_GRAPH"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print("Testing SGLang minimal setup...")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except Exception as e:
    print(f"✗ PyTorch error: {e}")
    sys.exit(1)

try:
    # Try to import SGLang
    import sglang
    print("✓ SGLang imported")

    # Try to create a simple runtime
    from sglang import function, gen, set_default_backend, Runtime

    print("Creating runtime with minimal settings...")
    runtime = Runtime(
        model_path="Qwen/Qwen3-32B-AWQ",
        quantization="awq",
        max_total_tokens=512,
        mem_fraction_static=0.5,
        trust_remote_code=True,
        disable_cuda_graph=True,
        disable_custom_all_reduce=True,
        disable_flashinfer=True,
        disable_radix_cache=True,
        attention_backend="torch_native"
    )

    print("✓ Runtime created")

    # Simple test
    @function
    def test(s):
        s += "Hello, " + gen("output", max_tokens=5)

    state = test.run()
    print(f"✓ Test output: {state['output']}")

except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Runtime error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete.")