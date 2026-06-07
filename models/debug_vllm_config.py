import sys
import os
# Run the patcher first to make sure everything is patched in this python process
sys.path.insert(0, '/models')
import patch_vllm

from vllm.config import ModelConfig
try:
    model_path = '/root/.cache/huggingface/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf'
    m_config = ModelConfig(
        model=model_path,
        tokenizer='Qwen/Qwen3.6-35B-A3B',
        tokenizer_mode='auto',
        trust_remote_code=False,
        dtype='float16',
        seed=0,
        revision=None,
    )
    print("Resolved architecture:", m_config.architecture)
    print("Model config architectures:", m_config.hf_config.architectures)
    print("Model type:", m_config.hf_config.model_type)
    print("Is MOE:", getattr(m_config.hf_config, "is_moe", False) or getattr(m_config.hf_text_config, "is_moe", False))
except Exception as e:
    import traceback
    traceback.print_exc()
