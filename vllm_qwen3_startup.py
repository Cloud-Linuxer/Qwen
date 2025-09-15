#!/usr/bin/env python3
import sys
import os

# Add Qwen3Next model registration
def register_qwen3next():
    try:
        from vllm.model_executor.models import ModelRegistry
        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

        # Register Qwen3NextForCausalLM as an alias to Qwen2ForCausalLM
        # since they share the same architecture
        ModelRegistry.register_model("Qwen3NextForCausalLM", Qwen2ForCausalLM)
        print("Successfully registered Qwen3NextForCausalLM model")

    except Exception as e:
        print(f"Warning: Could not register Qwen3Next model: {e}")
        # Try alternative registration method
        try:
            import vllm.model_executor.models.registry as registry
            from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

            # Directly add to the model registry
            if hasattr(registry, '_MODELS'):
                registry._MODELS["Qwen3NextForCausalLM"] = Qwen2ForCausalLM
                print("Successfully registered Qwen3NextForCausalLM using alternative method")
        except Exception as e2:
            print(f"Failed to register model: {e2}")

if __name__ == "__main__":
    register_qwen3next()

    # Now run the vLLM server
    from vllm.entrypoints.openai.api_server import run_server
    import asyncio
    from vllm.entrypoints.openai.arg_utils import make_arg_parser

    parser = make_arg_parser()
    args = parser.parse_args()

    # Import and run the main server
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    uvloop.run(run_server(args))
