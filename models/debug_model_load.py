import sys
import os

# Run the patcher first to apply all vLLM patches
sys.path.insert(0, '/models')
import patch_vllm

import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM

model_path = '/root/.cache/huggingface/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf'

try:
    print("Creating configs using EngineArgs...")
    engine_args = EngineArgs(
        model=model_path,
        tokenizer='Qwen/Qwen3.6-35B-A3B',
        quantization='gguf',
        dtype='float16',
        tensor_parallel_size=1,
        max_model_len=4096,
        enforce_eager=True,
    )
    vllm_config = engine_args.create_engine_config()

    from vllm.config.vllm import set_current_vllm_config
    with set_current_vllm_config(vllm_config):
        # Initialize single-process distributed state for metadata initialization
        from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        init_distributed_environment(backend="gloo")
        initialize_model_parallel(1, 1)

        print("Instantiating Qwen3_5MoeForCausalLM on meta device...")
        with torch.device("meta"):
            model = Qwen3_5MoeForCausalLM(vllm_config=vllm_config)
        
        print("\nModel instantiated successfully! Inspecting layer 0 linear_attn:")
        linear_attn = model.model.layers[0].linear_attn
        print("linear_attn type:", type(linear_attn))
        
        in_proj_qkvz = linear_attn.in_proj_qkvz
        print("in_proj_qkvz type:", type(in_proj_qkvz))
        print("in_proj_qkvz output_partition_sizes:", in_proj_qkvz.output_partition_sizes)
        
        print("\nParameters in in_proj_qkvz:")
        for name, param in in_proj_qkvz.named_parameters():
            loader = getattr(param, 'weight_loader', None)
            print(f"  Parameter: {name}")
            print(f"    is_gguf_weight: {getattr(param, 'is_gguf_weight', False)}")
            print(f"    is_gguf_weight_type: {getattr(param, 'is_gguf_weight_type', False)}")
            print(f"    weight_loader: {loader}")
            
        print("\nChecking if shared_expert and shared_expert_gate exist in layer 0:")
        mlp = model.model.layers[0].mlp
        print("mlp type:", type(mlp))
        print("hasattr(mlp, 'shared_expert'):", hasattr(mlp, 'shared_expert'))
        print("hasattr(mlp, 'shared_expert_gate'):", hasattr(mlp, 'shared_expert_gate'))
        if hasattr(mlp, 'shared_expert') and mlp.shared_expert is not None:
            print("shared_expert type:", type(mlp.shared_expert))
            print("Parameters in shared_expert:")
            for name, param in mlp.shared_expert.named_parameters():
                try:
                    shape = str(param.shape)
                except Exception:
                    shape = "Uninitialized"
                print(f"  {name}: {shape} (type: {type(param).__name__})")
        if hasattr(mlp, 'shared_expert_gate') and mlp.shared_expert_gate is not None:
            print("shared_expert_gate parameters:")
            for name, param in mlp.shared_expert_gate.named_parameters():
                try:
                    shape = str(param.shape)
                except Exception:
                    shape = "Uninitialized"
                print(f"  {name}: {shape} (type: {type(param).__name__})")

except Exception as e:
    import traceback
    traceback.print_exc()
