import gguf
import sys

try:
    reader = gguf.GGUFReader('/root/.cache/huggingface/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf')
    print("GGUF Tensors:")
    for tensor in reader.tensors:
        print(f"Name: {tensor.name}, Shape: {tensor.shape}, Type: {tensor.tensor_type.name}")
except Exception as e:
    import traceback
    traceback.print_exc()


