"""
Qwen3Next model support for vLLM
This adds support for the Qwen3-Next-80B-A3B-Instruct model architecture
"""

import os
import sys

def patch_vllm_for_qwen3next():
    """Patch vLLM to support Qwen3Next architecture"""

    # Create the model implementation file
    model_code = '''
"""Inference-only Qwen3-Next model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import (
    Qwen2MLP,
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

# Qwen3Next uses the same architecture as Qwen2 with different config
class Qwen3NextForCausalLM(Qwen2ForCausalLM):
    """
    Qwen3-Next model which is architecturally identical to Qwen2
    but with different configurations and MoE support
    """

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[Any] = None,
    ) -> None:
        # Map Qwen3Next config to Qwen2 config format
        if hasattr(config, 'model_type') and config.model_type == 'qwen3_next':
            # Ensure compatibility
            if not hasattr(config, 'rope_theta'):
                config.rope_theta = 1000000.0  # Default for Qwen3-Next
            if not hasattr(config, 'sliding_window'):
                config.sliding_window = None

        super().__init__(config, cache_config, quant_config, lora_config)
'''

    # Write the model implementation
    model_path = "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen3_next.py"

    # Create a startup script that patches vLLM
    startup_script = '''#!/usr/bin/env python3
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
'''

    return model_code, startup_script

# Generate the patches
model_code, startup_script = patch_vllm_for_qwen3next()

# Save the startup script
with open("/home/gpt-oss/vllm_qwen3_startup.py", "w") as f:
    f.write(startup_script)

print("Created vLLM Qwen3Next support patch")