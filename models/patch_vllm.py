import os
import sys

# 1. Patch transformers modeling_gguf_pytorch_utils.py
file_path_trans = '/usr/local/lib/python3.12/dist-packages/transformers/modeling_gguf_pytorch_utils.py'
if os.path.exists(file_path_trans):
    try:
        with open(file_path_trans, 'r', encoding='utf-8') as f:
            content = f.read()

        changed = False

        # TENSOR_PROCESSORS patch
        target_tensor = '"qwen3moe": Qwen2MoeTensorProcessor,'
        replacement_tensor = '"qwen3moe": Qwen2MoeTensorProcessor,\n    "qwen35moe": Qwen2MoeTensorProcessor,'
        if target_tensor in content and '"qwen35moe"' not in content:
            content = content.replace(target_tensor, replacement_tensor)
            changed = True
            print("[Patch] Added qwen35moe to TENSOR_PROCESSORS")

        # model_type mapping patch
        # Clean up previous qwen3_5_moe patch first
        if 'elif model_type == "qwen3_5_moe":' in content:
            content = content.replace(
                'elif model_type == "qwen3_5_moe":\n        model_type = "qwen35moe"',
                'elif model_type == "qwen3_5_moe_text":\n        model_type = "qwen35moe"'
            )
            changed = True
            print("[Patch] Updated model_type mapping to qwen3_5_moe_text")
        else:
            target_type = 'elif model_type == "qwen3_moe":\n        model_type = "qwen3moe"'
            replacement_type = 'elif model_type == "qwen3_moe":\n        model_type = "qwen3moe"\n    elif model_type == "qwen3_5_moe_text":\n        model_type = "qwen35moe"'
            if target_type in content and '"qwen3_5_moe_text"' not in content:
                content = content.replace(target_type, replacement_type)
                changed = True
                print("[Patch] Added qwen3_5_moe_text to model_type mapping")

        # Architecture mapping patch
        # Handle cleanup of previous qwen3_5_moe patch, patched qwen3_moe, and fresh config
        target_arch_prev = 'elif "qwen35moe" in architecture:\n        updated_architecture = "qwen3_5_moe"'
        target_arch_patched = 'elif "qwen3moe" in architecture or "qwen35moe" in architecture:\n        updated_architecture = "qwen3_moe"'
        target_arch_fresh = 'elif "qwen3moe" in architecture:\n        updated_architecture = "qwen3_moe"'
        replacement_arch = 'elif "qwen35moe" in architecture:\n        updated_architecture = "qwen3_5_moe_text"\n    elif "qwen3moe" in architecture:\n        updated_architecture = "qwen3_moe"'
        
        if target_arch_prev in content:
            content = content.replace(target_arch_prev, 'elif "qwen35moe" in architecture:\n        updated_architecture = "qwen3_5_moe_text"')
            changed = True
            print("[Patch] Updated updated_architecture mapping to qwen3_5_moe_text")
        elif target_arch_patched in content:
            content = content.replace(target_arch_patched, replacement_arch)
            changed = True
            print("[Patch] Mapped qwen35moe to qwen3_5_moe_text (replaced previous patch)")
        elif target_arch_fresh in content and 'updated_architecture = "qwen3_5_moe_text"' not in content:
            content = content.replace(target_arch_fresh, replacement_arch)
            changed = True
            print("[Patch] Mapped qwen35moe to qwen3_5_moe_text (fresh patch)")

        # GGUF_TO_TRANSFORMERS_MAPPING and GGUF_SUPPORTED_ARCHITECTURES patch
        # Clean up old qwen3_5_moe block if exists
        if 'GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3_5_moe"]' in content:
            content = content.replace(
                'GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3_5_moe"] = GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3_moe"]\nif "qwen3_5_moe" not in GGUF_SUPPORTED_ARCHITECTURES:\n    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3_5_moe")',
                'GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3_5_moe_text"] = GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3_moe"]\nif "qwen3_5_moe_text" not in GGUF_SUPPORTED_ARCHITECTURES:\n    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3_5_moe_text")'
            )
            changed = True
            print("[Patch] Updated GGUF_TO_TRANSFORMERS_MAPPING additions to qwen3_5_moe_text")
        elif 'GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3_5_moe_text"]' not in content:
            content += '\n\n# Qwen3.5 MoE patch\nGGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3_5_moe_text"] = GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3_moe"]\nif "qwen3_5_moe_text" not in GGUF_SUPPORTED_ARCHITECTURES:\n    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3_5_moe_text")\n'
            changed = True
            print("[Patch] Appended GGUF_TO_TRANSFORMERS_MAPPING and GGUF_SUPPORTED_ARCHITECTURES additions")

        if changed:
            with open(file_path_trans, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[Patch] Successfully patched modeling_gguf_pytorch_utils.py!")
        else:
            print("[Patch] modeling_gguf_pytorch_utils.py patches already applied.")
    except Exception as e:
        print(f"[Patch] Error patching transformers: {e}")
else:
    print(f"[Patch] Transformers file not found at {file_path_trans}, skipping patch.")

# 2. Patch vllm registry.py
file_path_registry = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py'
if os.path.exists(file_path_registry):
    try:
        with open(file_path_registry, 'r', encoding='utf-8') as f:
            content = f.read()

        target = '"Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),'
        replacement = '"Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),\n    "Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),\n    "Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),'

        if target in content and '"Qwen3_5MoeForCausalLM"' not in content:
            content = content.replace(target, replacement)
            with open(file_path_registry, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[Patch] Successfully registered Qwen3_5ForCausalLM and Qwen3_5MoeForCausalLM in registry.py!")
        else:
            print("[Patch] Qwen3.5 registrations in registry.py already applied.")
    except Exception as e:
        print(f"[Patch] Error patching registry.py: {e}")
else:
    print(f"[Patch] registry.py not found at {file_path_registry}")

# 3. Patch vllm gguf_loader.py
file_path_vllm = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py'
if os.path.exists(file_path_vllm):
    try:
        with open(file_path_vllm, 'r', encoding='utf-8') as f:
            content = f.read()

        changed = False

        # Update the expert mapping model_type condition check
        # Clean up old qwen3_5_moe patch if exists
        if 'if model_type in ("qwen2_moe", "qwen3_moe", "qwen3_5_moe"):' in content:
            content = content.replace(
                'if model_type in ("qwen2_moe", "qwen3_moe", "qwen3_5_moe"):',
                'if model_type in ("qwen2_moe", "qwen3_moe", "qwen3_5_moe_text"):'
            )
            changed = True
            print("[Patch] Updated model_type condition from qwen3_5_moe to qwen3_5_moe_text in gguf_loader.py")
        else:
            target_check_fresh = 'if model_type in ("qwen2_moe", "qwen3_moe"):'
            replacement_check = 'if model_type in ("qwen2_moe", "qwen3_moe", "qwen3_5_moe_text"):'
            if target_check_fresh in content:
                content = content.replace(target_check_fresh, replacement_check)
                changed = True
                print("[Patch] Updated model_type condition to include qwen3_5_moe_text in gguf_loader.py")

        # Map qwen3_5_moe_text model_type to qwen35moe to match GGUF architecture names
        target_replace = '        if model_type in ("qwen2_moe", "qwen3_moe", "qwen3_5_moe_text"):\n            model_type = model_type.replace("_", "")'
        replacement_replace = '        if model_type in ("qwen2_moe", "qwen3_moe", "qwen3_5_moe_text"):\n            if model_type == "qwen3_5_moe_text":\n                model_type = "qwen35moe"\n            else:\n                model_type = model_type.replace("_", "")'
        if target_replace in content:
            content = content.replace(target_replace, replacement_replace)
            changed = True
            print("[Patch] Mapped qwen3_5_moe_text model_type to qwen35moe in gguf_loader.py")

        # Map dt_bias manually in gguf_loader.py
        target_expert_map = (
            '                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (\n'
            '                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"\n'
            '                )'
        )
        replacement_expert_map = (
            '                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (\n'
            '                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"\n'
            '                )\n'
            '                if hasattr(config, "layer_types") and idx < len(config.layer_types):\n'
            '                    if config.layer_types[idx] == "linear_attention":\n'
            '                        gguf_to_hf_name_map[f"blk.{idx}.ssm_dt.bias"] = (\n'
            '                            f"model.layers.{idx}.linear_attn.dt_bias"\n'
            '                        )'
        )
        if target_expert_map in content and 'linear_attn.dt_bias' not in content:
            content = content.replace(target_expert_map, replacement_expert_map)
            changed = True
            print("[Patch] Added linear_attn.dt_bias manual mapping in gguf_loader.py")

        # Fix trailing dot bug when suffix is empty in gguf_loader.py
        target_dot = (
            '            if gguf_name is None:\n'
            '                return None\n\n'
            '            return gguf_name + "." + suffix'
        )
        replacement_dot = (
            '            if gguf_name is None:\n'
            '                return None\n\n'
            '            return gguf_name + ("." + suffix if suffix else "")'
        )
        if target_dot in content:
            content = content.replace(target_dot, replacement_dot)
            changed = True
            print("[Patch] Fixed trailing dot bug in gguf_loader.py")

        # Revert the custom is_multimodal checks we added previously (since Qwen3_5MoeTextConfig has no vision_config, is_multimodal will naturally be False)
        target_im1_patched = (
            '        is_multimodal = (\n'
            '            hasattr(config, "vision_config") and config.vision_config is not None\n'
            '            and getattr(model_config, "multimodal_config", None) is not None\n'
            '        )'
        )
        replacement_im1_orig = (
            '        is_multimodal = (\n'
            '            hasattr(config, "vision_config") and config.vision_config is not None\n'
            '        )'
        )
        if target_im1_patched in content:
            content = content.replace(target_im1_patched, replacement_im1_orig)
            changed = True
            print("[Patch] Reverted is_multimodal check 1 to original")

        target_im2_patched = 'is_multimodal = hasattr(model_config.hf_config, "vision_config") and getattr(model_config, "multimodal_config", None) is not None'
        replacement_im2_orig = 'is_multimodal = hasattr(model_config.hf_config, "vision_config")'
        if target_im2_patched in content:
            content = content.replace(target_im2_patched, replacement_im2_orig)
            changed = True
            print("[Patch] Reverted is_multimodal check 2 to original")

        target_im3_patched = 'is_multimodal = hasattr(hf_config, "vision_config") and getattr(model_config, "multimodal_config", None) is not None'
        replacement_im3_orig = 'is_multimodal = hasattr(hf_config, "vision_config")'
        if target_im3_patched in content:
            content = content.replace(target_im3_patched, replacement_im3_orig)
            changed = True
            print("[Patch] Reverted is_multimodal check 3 to original")

        # Force lm_head and shared_expert_gate to be in unquantized_modules and store on self
        old_patched_1 = (
            'vllm_config.quant_config.unquantized_modules.extend(unquant_names)\n'
            '        if "lm_head" not in vllm_config.quant_config.unquantized_modules:\n'
            '            vllm_config.quant_config.unquantized_modules.append("lm_head")\n'
            '        self.unquantized_modules = vllm_config.quant_config.unquantized_modules'
        )
        old_patched_2 = (
            'vllm_config.quant_config.unquantized_modules.extend(unquant_names)\n'
            '        if "lm_head" not in vllm_config.quant_config.unquantized_modules:\n'
            '            vllm_config.quant_config.unquantized_modules.append("lm_head")\n'
            '        if "shared_expert_gate" not in vllm_config.quant_config.unquantized_modules:\n'
            '            vllm_config.quant_config.unquantized_modules.append("shared_expert_gate")\n'
            '        self.unquantized_modules = vllm_config.quant_config.unquantized_modules'
        )
        if old_patched_2 in content:
            content = content.replace(old_patched_2, 'vllm_config.quant_config.unquantized_modules.extend(unquant_names)')
            changed = True
        elif old_patched_1 in content:
            content = content.replace(old_patched_1, 'vllm_config.quant_config.unquantized_modules.extend(unquant_names)')
            changed = True

        target_unquant = 'vllm_config.quant_config.unquantized_modules.extend(unquant_names)'
        replacement_unquant = (
            'vllm_config.quant_config.unquantized_modules.extend(unquant_names)\n'
            '        if "lm_head" not in vllm_config.quant_config.unquantized_modules:\n'
            '            vllm_config.quant_config.unquantized_modules.append("lm_head")\n'
            '        if "shared_expert_gate" not in vllm_config.quant_config.unquantized_modules:\n'
            '            vllm_config.quant_config.unquantized_modules.append("shared_expert_gate")\n'
            '        self.unquantized_modules = vllm_config.quant_config.unquantized_modules'
        )
        if target_unquant in content:
            content = content.replace(target_unquant, replacement_unquant)
            changed = True
            print("[Patch] Appended lm_head and shared_expert_gate to unquantized_modules and stored on self")

        # Patch _get_weights_iterator calls to pass unquantized_modules
        if 'getattr(self, "unquantized_modules", None)' not in content:
            target_iterator_single = 'yield from gguf_quant_weights_iterator(mmproj_file, gguf_to_hf_name_map)'
            replacement_iterator_single = 'yield from gguf_quant_weights_iterator(mmproj_file, gguf_to_hf_name_map, getattr(self, "unquantized_modules", None))'
            if target_iterator_single in content:
                content = content.replace(target_iterator_single, replacement_iterator_single)
                changed = True
                print("[Patch] Passed unquantized_modules to mmproj gguf_quant_weights_iterator in gguf_loader.py")

            target_iterator_multi_shards = 'yield from gguf_quant_weights_iterator_multi(\n                gguf_files, gguf_to_hf_name_map\n            )'
            replacement_iterator_multi_shards = 'yield from gguf_quant_weights_iterator_multi(\n                gguf_files, gguf_to_hf_name_map, getattr(self, "unquantized_modules", None)\n            )'
            if target_iterator_multi_shards in content:
                content = content.replace(target_iterator_multi_shards, replacement_iterator_multi_shards)
                changed = True
                print("[Patch] Passed unquantized_modules to multi-shards iterator in gguf_loader.py")

            target_iterator_single_shard = 'yield from gguf_quant_weights_iterator(\n                model_name_or_path, gguf_to_hf_name_map\n            )'
            replacement_iterator_single_shard = 'yield from gguf_quant_weights_iterator(\n                model_name_or_path, gguf_to_hf_name_map, getattr(self, "unquantized_modules", None)\n            )'
            if target_iterator_single_shard in content:
                content = content.replace(target_iterator_single_shard, replacement_iterator_single_shard)
                changed = True
                print("[Patch] Passed unquantized_modules to single-shard iterator in gguf_loader.py")

        # Clean up any old broken sideload patches
        broken_sideload_1 = (
            '                sideload_params.append(\n'
            '                    re.compile(\n'
            '                        f"model\\\\.layers\\\\.{idx}"\n'
            '                        r"\\.mlp\\.experts\\.gate_up_proj"\n'
            '                    )\n'
            '                )\n'
        )
        if broken_sideload_1 in content:
            content = content.replace(broken_sideload_1, '')
            changed = True

        # Target using a unique, exact line mapping
        target_up_exps = (
            '                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (\n'
            '                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"\n'
            '                )'
        )
        if target_up_exps in content and 'ffn_down_shexp.weight' not in content:
            replacement_up_exps = (
                '                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (\n'
                '                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"\n'
                '                )\n'
                '                sideload_params.append(\n'
                '                    re.compile(\n'
                '                        f"model\\\\.layers\\\\.{idx}"\n'
                '                        r"\\.mlp\\.experts\\.gate_up_proj"\n'
                '                    )\n'
                '                )\n'
                '                if model_type == "qwen35moe":\n'
                '                    gguf_to_hf_name_map[f"blk.{idx}.ffn_down_shexp.weight"] = (\n'
                '                        f"model.layers.{idx}.mlp.shared_expert.down_proj.weight"\n'
                '                    )\n'
                '                    gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_shexp.weight"] = (\n'
                '                        f"model.layers.{idx}.mlp.shared_expert.gate_proj.weight"\n'
                '                    )\n'
                '                    gguf_to_hf_name_map[f"blk.{idx}.ffn_up_shexp.weight"] = (\n'
                '                        f"model.layers.{idx}.mlp.shared_expert.up_proj.weight"\n'
                '                    )\n'
                '                    gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_inp_shexp.weight"] = (\n'
                '                        f"model.layers.{idx}.mlp.shared_expert_gate.weight"\n'
                '                    )'
            )
            content = content.replace(target_up_exps, replacement_up_exps)
            changed = True
            print("[Patch] Added gate_up_proj sideload and shared expert mappings to gguf_loader.py")

        if changed:
            with open(file_path_vllm, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[Patch] Successfully patched gguf_loader.py!")
        else:
            print("[Patch] gguf_loader.py patches already applied.")
    except Exception as e:
        print(f"[Patch] Error patching vllm: {e}")
else:
    print(f"[Patch] vLLM gguf_loader.py file not found at {file_path_vllm}, skipping patch.")

# 4. Patch fused_moe/layer.py
file_path_layer = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py'
if os.path.exists(file_path_layer):
    try:
        with open(file_path_layer, 'r', encoding='utf-8') as f:
            content = f.read()

        target = '                param = getattr(self, param_name)'
        replacement = '                if not hasattr(self, param_name) and hasattr(self, param_name.replace("_weight", "_qweight")):\n                    param_name = param_name.replace("_weight", "_qweight")\n                param = getattr(self, param_name)'

        if target in content and 'param_name.replace("_weight", "_qweight")' not in content:
            content = content.replace(target, replacement)
            with open(file_path_layer, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[Patch] Successfully patched FusedMoE layer.py for qweight compatibility!")
        else:
            print("[Patch] layer.py patch already applied.")
    except Exception as e:
        print(f"[Patch] Error patching layer.py: {e}")
else:
    print(f"[Patch] layer.py not found at {file_path_layer}")

# 5. Patch vllm qwen3_5.py
file_path_qwen3_5 = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py'
if os.path.exists(file_path_qwen3_5):
    try:
        with open(file_path_qwen3_5, 'r', encoding='utf-8') as f:
            content = f.read()

        content = content.replace('\r\n', '\n')
        changed = False

        # 5a. Add weight processing logging print
        target0 = (
            '        for name, loaded_weight in weights:\n'
            '            if "rotary_emb.inv_freq" in name:'
        )
        replacement0 = (
            '        for name, loaded_weight in weights:\n'
            '            print("[Qwen-Load] Processing checkpoint weight name:", name)\n'
            '            if "rotary_emb.inv_freq" in name:'
        )
        if target0 in content and '[Qwen-Load]' not in content:
            content = content.replace(target0, replacement0)
            changed = True
            print("[Patch] Added weight name printing in qwen3_5.py")

        # 5b. Remap .weight to .qweight for stacked_params_mapping
        target1 = (
            '                # name = apply_attn_prefix(name, params_dict)\n'
            '                if name not in params_dict:\n'
            '                    continue'
        )
        replacement1 = (
            '                # name = apply_attn_prefix(name, params_dict)\n'
            '                if name not in params_dict and name.replace(".weight", ".qweight") in params_dict:\n'
            '                    name = name.replace(".weight", ".qweight")\n'
            '                if name not in params_dict:\n'
            '                    continue'
        )
        if target1 in content and 'name.replace(".weight", ".qweight") in params_dict' not in content:
            content = content.replace(target1, replacement1)
            changed = True
            print("[Patch] Added qweight fallback for stacked parameters in qwen3_5.py")

        # 5c. Remap .weight to .qweight for expert_params_mapping
        target2 = (
            '                    # Skip layers on other devices.\n'
            '                    if is_pp_missing_parameter(name_mapped, self):\n'
            '                        continue\n'
            '                    if is_fused_expert:'
        )
        replacement2 = (
            '                    # Skip layers on other devices.\n'
            '                    if is_pp_missing_parameter(name_mapped, self):\n'
            '                        continue\n'
            '                    if name_mapped not in params_dict and name_mapped.replace(".weight", ".qweight") in params_dict:\n'
            '                        name_mapped = name_mapped.replace(".weight", ".qweight")\n'
            '                    if is_fused_expert:'
        )
        if target2 in content and 'name_mapped.replace(".weight", ".qweight") in params_dict' not in content:
            content = content.replace(target2, replacement2)
            changed = True
            print("[Patch] Added qweight fallback for expert parameters in qwen3_5.py")

        # 5d. Remap .weight to .qweight for default weight loading block
        target3 = (
            '                    if is_pp_missing_parameter(name, self):\n'
            '                        continue\n'
            '                    if name not in params_dict:\n'
            '                        logger.warning_once('
        )
        replacement3 = (
            '                    if is_pp_missing_parameter(name, self):\n'
            '                        continue\n'
            '                    if name not in params_dict and name.replace(".weight", ".qweight") in params_dict:\n'
            '                        name = name.replace(".weight", ".qweight")\n'
            '                    if name not in params_dict:\n'
            '                        logger.warning_once('
        )
        if target3 in content and 'name.replace(".weight", ".qweight") in params_dict' not in content:
            content = content.replace(target3, replacement3)
            changed = True
            print("[Patch] Added qweight fallback for fallback parameters in qwen3_5.py")

        # 5e. Print uninitialized parameters check at the end of load_weights
        target4 = (
            '            loaded_params.add(name)\n'
            '        return loaded_params'
        )
        if '[GGUF-Init-Check]' in content and 'current_params = dict(self.named_parameters())' not in content:
            old_patch_str = (
                '        from torch.nn.parameter import UninitializedParameter\n'
                '        uninit = [p_name for p_name, param in params_dict.items() if isinstance(param, UninitializedParameter)]\n'
                '        if uninit:\n'
                '            print("[GGUF-Init-Check] Found uninitialized parameters:", uninit)\n'
                '        else:\n'
                '            print("[GGUF-Init-Check] All parameters successfully initialized!")'
            )
            new_patch_str = (
                '        from torch.nn.parameter import UninitializedParameter\n'
                '        current_params = dict(self.named_parameters())\n'
                '        uninit = [p_name for p_name, param in current_params.items() if isinstance(param, UninitializedParameter)]\n'
                '        if uninit:\n'
                '            print("[GGUF-Init-Check] Found uninitialized parameters:", uninit)\n'
                '        else:\n'
                '            print("[GGUF-Init-Check] All parameters successfully initialized!")'
            )
            content = content.replace(old_patch_str, new_patch_str)
            changed = True
            print("[Patch] Updated GGUF-Init-Check to query named_parameters() dynamically")
        elif '[GGUF-Init-Check]' not in content:
            replacement4 = (
                '            loaded_params.add(name)\n'
                '        from torch.nn.parameter import UninitializedParameter\n'
                '        current_params = dict(self.named_parameters())\n'
                '        uninit = [p_name for p_name, param in current_params.items() if isinstance(param, UninitializedParameter)]\n'
                '        if uninit:\n'
                '            print("[GGUF-Init-Check] Found uninitialized parameters:", uninit)\n'
                '        else:\n'
                '            print("[GGUF-Init-Check] All parameters successfully initialized!")\n'
                '        return loaded_params'
            )
            content = content.replace(target4, replacement4)
            changed = True
            print("[Patch] Added dynamic uninitialized parameters check in qwen3_5.py")

        # 5f. Add quant_config and prefix to VocabParallelEmbedding
        target5 = (
            '        self.embed_tokens = VocabParallelEmbedding(\n'
            '            self.vocab_size,\n'
            '            config.hidden_size,\n'
            '        )'
        )
        replacement5 = (
            '        self.embed_tokens = VocabParallelEmbedding(\n'
            '            self.vocab_size,\n'
            '            config.hidden_size,\n'
            '            quant_config=vllm_config.quant_config,\n'
            '            prefix=f"{prefix}.embed_tokens",\n'
            '        )'
        )
        if target5 in content and 'quant_config=vllm_config.quant_config' not in content:
            content = content.replace(target5, replacement5)
            changed = True
            print("[Patch] Added quant_config and prefix to VocabParallelEmbedding in qwen3_5.py")

        # 5g. Add quant_config to ParallelLMHead
        target6 = (
            '            else:\n'
            '                self.lm_head = ParallelLMHead(\n'
            '                    config.vocab_size,\n'
            '                    config.hidden_size,\n'
            '                    prefix=maybe_prefix(prefix, "lm_head"),\n'
            '                )'
        )
        replacement6 = (
            '            else:\n'
            '                self.lm_head = ParallelLMHead(\n'
            '                    config.vocab_size,\n'
            '                    config.hidden_size,\n'
            '                    quant_config=vllm_config.quant_config,\n'
            '                    prefix=maybe_prefix(prefix, "lm_head"),\n'
            '                )'
        )
        if target6 in content and 'quant_config=vllm_config.quant_config' not in content:
            content = content.replace(target6, replacement6)
            changed = True
            print("[Patch] Added quant_config to ParallelLMHead in qwen3_5.py")

        if changed:
            with open(file_path_qwen3_5, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[Patch] Successfully patched qwen3_5.py!")
        else:
            print("[Patch] qwen3_5.py patches already applied.")
    except Exception as e:
        print(f"[Patch] Error patching qwen3_5: {e}")
else:
    print(f"[Patch] qwen3_5.py file not found at {file_path_qwen3_5}, skipping patch.")

# 6. Patch vllm weight_utils.py
file_path_wu = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py'
if os.path.exists(file_path_wu):
    try:
        with open(file_path_wu, 'r', encoding='utf-8') as f:
            content = f.read()

        changed = False

        # Self-healing: if the broken patch with local imports is present, clean it up
        broken_str = '                    import gguf\n                    import numpy as np\n'
        if broken_str in content:
            content = content.replace(broken_str, '')
            changed = True
            print("[Patch] Cleaned up broken local imports in weight_utils.py")

        if 'unquantized_modules: list[str] = None' not in content:
            target_single = '''def gguf_quant_weights_iterator(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """
    Iterate over the quant weights in the model gguf files and convert
    them to torch tensors.
    Be careful of the order of yielding weight types and weights data,
    we have to yield all weight types first before yielding any weights.
    Otherwise it would cause issue when loading weights with for packed
    layer with different quant types.
    """

    reader = gguf.GGUFReader(gguf_file)

    for tensor in reader.tensors:
        if tensor.name in gguf_to_hf_name_map:
            weight_type = tensor.tensor_type
            name = gguf_to_hf_name_map[tensor.name]

            if weight_type.name not in ("F32", "BF16", "F16"):
                weight_type_name = name.replace("weight", "qweight_type")
                weight_type = torch.tensor(weight_type)
                yield weight_type_name, weight_type

    for tensor in reader.tensors:
        if tensor.name in gguf_to_hf_name_map:
            weight = tensor.data
            weight_type = tensor.tensor_type
            name = gguf_to_hf_name_map[tensor.name]
            if weight_type.name not in ("F32", "BF16", "F16"):
                name = name.replace("weight", "qweight")
            if weight_type.name == "BF16" and tensor.data.dtype == np.uint8:
                # BF16 is currently the only "quantization" type that isn't
                # actually quantized but is read as a raw byte tensor.
                # Reinterpret as `torch.bfloat16` tensor.
                weight = weight.view(np.uint16)
                if reader.byte_order == "S":
                    # GGUF endianness != system endianness
                    weight = weight.byteswap()
                param = torch.tensor(weight).view(torch.bfloat16)
            else:
                param = torch.tensor(weight)
            yield name, param'''

            replacement_single = '''def gguf_quant_weights_iterator(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str], unquantized_modules: list[str] = None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """
    Iterate over the quant weights in the model gguf files and convert
    them to torch tensors.
    Be careful of the order of yielding weight types and weights data,
    we have to yield all weight types first before yielding any weights.
    Otherwise it would cause issue when loading weights with for packed
    layer with different quant types.
    """

    reader = gguf.GGUFReader(gguf_file)

    for tensor in reader.tensors:
        if tensor.name in gguf_to_hf_name_map:
            weight_type = tensor.tensor_type
            name = gguf_to_hf_name_map[tensor.name]

            if weight_type.name not in ("F32", "BF16", "F16"):
                if unquantized_modules is not None and any(m in name for m in unquantized_modules):
                    continue
                weight_type_name = name.replace("weight", "qweight_type")
                weight_type = torch.tensor(weight_type)
                yield weight_type_name, weight_type

    for tensor in reader.tensors:
        if tensor.name in gguf_to_hf_name_map:
            weight = tensor.data
            weight_type = tensor.tensor_type
            name = gguf_to_hf_name_map[tensor.name]
            is_unquantized_module = (unquantized_modules is not None and any(m in name for m in unquantized_modules))
            if weight_type.name not in ("F32", "BF16", "F16"):
                if is_unquantized_module:
                    dequantized_data = gguf.dequantize(tensor.data, tensor.tensor_type)
                    param = torch.from_numpy(dequantized_data.copy())
                    if "ffn_gate_inp_shexp" in tensor.name or "shared_expert_gate" in name:
                        if param.ndim == 1:
                            param = param.unsqueeze(0)
                    elif "ssm_conv1d" in tensor.name or "conv1d.weight" in name:
                        if param.ndim == 2:
                            param = param.unsqueeze(1)
                    elif "ssm_a" in tensor.name or name.endswith(".A"):
                        param = torch.log(-param)
                    yield name, param
                    continue
                else:
                    name = name.replace("weight", "qweight")
            if weight_type.name == "BF16" and tensor.data.dtype == np.uint8:
                # BF16 is currently the only "quantization" type that isn't
                # actually quantized but is read as a raw byte tensor.
                # Reinterpret as `torch.bfloat16` tensor.
                weight = weight.view(np.uint16)
                if reader.byte_order == "S":
                    # GGUF endianness != system endianness
                    weight = weight.byteswap()
                param = torch.tensor(weight).view(torch.bfloat16)
            else:
                param = torch.tensor(weight)
            if "ffn_gate_inp_shexp" in tensor.name or "shared_expert_gate" in name:
                if param.ndim == 1:
                    param = param.unsqueeze(0)
            elif "ssm_conv1d" in tensor.name or "conv1d.weight" in name:
                if param.ndim == 2:
                    param = param.unsqueeze(1)
            elif "ssm_a" in tensor.name or name.endswith(".A"):
                param = torch.log(-param)
            yield name, param'''

            target_multi = '''def gguf_quant_weights_iterator_multi(
    gguf_files: list[str], gguf_to_hf_name_map: dict[str, str]
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """
    Iterate over the quant weights across multiple GGUF shard files
    and convert them to torch tensors.

    Like gguf_quant_weights_iterator, we yield all weight types first
    before yielding any weights data to avoid issues with packed layers
    that have different quant types.
    """
    readers = [gguf.GGUFReader(f) for f in gguf_files]

    # First pass: yield all weight types across all shards
    for reader in readers:
        for tensor in reader.tensors:
            if tensor.name in gguf_to_hf_name_map:
                weight_type = tensor.tensor_type
                name = gguf_to_hf_name_map[tensor.name]
                if weight_type.name not in ("F32", "BF16", "F16"):
                    weight_type_name = name.replace("weight", "qweight_type")
                    weight_type = torch.tensor(weight_type)
                    yield weight_type_name, weight_type

    # Second pass: yield all weight data across all shards
    for reader in readers:
        for tensor in reader.tensors:
            if tensor.name in gguf_to_hf_name_map:
                weight = tensor.data
                weight_type = tensor.tensor_type
                name = gguf_to_hf_name_map[tensor.name]
                if weight_type.name not in ("F32", "BF16", "F16"):
                    name = name.replace("weight", "qweight")
                if weight_type.name == "BF16" and tensor.data.dtype == np.uint8:
                    weight = weight.view(np.uint16)
                    if reader.byte_order == "S":
                        weight = weight.byteswap()
                    param = torch.tensor(weight).view(torch.bfloat16)
                else:
                    param = torch.tensor(weight)
                yield name, param'''

            replacement_multi = '''def gguf_quant_weights_iterator_multi(
    gguf_files: list[str], gguf_to_hf_name_map: dict[str, str], unquantized_modules: list[str] = None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """
    Iterate over the quant weights across multiple GGUF shard files
    and convert them to torch tensors.

    Like gguf_quant_weights_iterator, we yield all weight types first
    before yielding any weights data to avoid issues with packed layers
    that have different quant types.
    """
    readers = [gguf.GGUFReader(f) for f in gguf_files]

    # First pass: yield all weight types across all shards
    for reader in readers:
        for tensor in reader.tensors:
            if tensor.name in gguf_to_hf_name_map:
                weight_type = tensor.tensor_type
                name = gguf_to_hf_name_map[tensor.name]
                if weight_type.name not in ("F32", "BF16", "F16"):
                    if unquantized_modules is not None and any(m in name for m in unquantized_modules):
                        continue
                    weight_type_name = name.replace("weight", "qweight_type")
                    weight_type = torch.tensor(weight_type)
                    yield weight_type_name, weight_type

    # Second pass: yield all weight data across all shards
    for reader in readers:
        for tensor in reader.tensors:
            if tensor.name in gguf_to_hf_name_map:
                weight = tensor.data
                weight_type = tensor.tensor_type
                name = gguf_to_hf_name_map[tensor.name]
                is_unquantized_module = (unquantized_modules is not None and any(m in name for m in unquantized_modules))
                if weight_type.name not in ("F32", "BF16", "F16"):
                    if is_unquantized_module:
                        dequantized_data = gguf.dequantize(tensor.data, tensor.tensor_type)
                        param = torch.from_numpy(dequantized_data.copy())
                        if "ffn_gate_inp_shexp" in tensor.name or "shared_expert_gate" in name:
                            if param.ndim == 1:
                                param = param.unsqueeze(0)
                        elif "ssm_conv1d" in tensor.name or "conv1d.weight" in name:
                            if param.ndim == 2:
                                param = param.unsqueeze(1)
                        elif "ssm_a" in tensor.name or name.endswith(".A"):
                            param = torch.log(-param)
                        yield name, param
                        continue
                    else:
                        name = name.replace("weight", "qweight")
                if weight_type.name == "BF16" and tensor.data.dtype == np.uint8:
                    weight = weight.view(np.uint16)
                    if reader.byte_order == "S":
                        weight = weight.byteswap()
                    param = torch.tensor(weight).view(torch.bfloat16)
                else:
                    param = torch.tensor(weight)
                if "ffn_gate_inp_shexp" in tensor.name or "shared_expert_gate" in name:
                    if param.ndim == 1:
                        param = param.unsqueeze(0)
                elif "ssm_conv1d" in tensor.name or "conv1d.weight" in name:
                    if param.ndim == 2:
                        param = param.unsqueeze(1)
                elif "ssm_a" in tensor.name or name.endswith(".A"):
                    param = torch.log(-param)
                yield name, param'''

            content = content.replace('\r\n', '\n')
            if target_single in content:
                content = content.replace(target_single, replacement_single)
                changed = True
                print("[Patch] Patched gguf_quant_weights_iterator in weight_utils.py")
            else:
                print("[Patch] WARNING: target_single not matched exactly in weight_utils.py")
                
            if target_multi in content:
                content = content.replace(target_multi, replacement_multi)
                changed = True
                print("[Patch] Patched gguf_quant_weights_iterator_multi in weight_utils.py")
            else:
                print("[Patch] WARNING: target_multi not matched exactly in weight_utils.py")

        if changed:
            with open(file_path_wu, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[Patch] Successfully patched weight_utils.py!")
        else:
            print("[Patch] weight_utils.py was not modified.")
    except Exception as e:
        print(f"[Patch] Error patching weight_utils.py: {e}")
else:
    print(f"[Patch] weight_utils.py not found at {file_path_wu}")

# 7. Patch vllm linear.py
file_path_lin = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/linear.py'
if os.path.exists(file_path_lin):
    try:
        with open(file_path_lin, 'r', encoding='utf-8') as f:
            content = f.read()

        changed = False

        target_lin = (
            '        is_gguf_weight = getattr(param, "is_gguf_weight", False)\n'
            '        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)\n'
            '        if isinstance(loaded_shard_id, tuple) and (\n'
            '            is_gguf_weight or is_gguf_weight_type\n'
            '        ):\n'
            '            raise NotImplementedError(\n'
            '                "Shard id with multiple indices is not supported for GGUF."\n'
            '            )\n'
            '        if is_gguf_weight_type:\n'
            '            if loaded_shard_id is not None:\n'
            '                param.data[loaded_shard_id].copy_(loaded_weight)\n'
            '                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()'
        )

        replacement_lin = (
            '        is_gguf_weight = getattr(param, "is_gguf_weight", False)\n'
            '        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)\n'
            '        if is_gguf_weight_type:\n'
            '            if loaded_shard_id is not None:\n'
            '                if isinstance(loaded_shard_id, tuple):\n'
            '                    param.data[list(loaded_shard_id)].copy_(loaded_weight)\n'
            '                    param.shard_weight_type[loaded_shard_id] = loaded_weight.item()\n'
            '                    for idx in loaded_shard_id:\n'
            '                        param.shard_weight_type[idx] = loaded_weight.item()\n'
            '                else:\n'
            '                    param.data[loaded_shard_id].copy_(loaded_weight)\n'
            '                    param.shard_weight_type[loaded_shard_id] = loaded_weight.item()'
        )

        content = content.replace('\r\n', '\n')
        if target_lin in content:
            content = content.replace(target_lin, replacement_lin)
            changed = True
            print("[Patch] Successfully patched weight_loader in linear.py for tuple shard support")
        else:
            if 'list(loaded_shard_id)' in content:
                old_subpatch = (
                    '                if isinstance(loaded_shard_id, tuple):\n'
                    '                    param.data[list(loaded_shard_id)].copy_(loaded_weight)\n'
                    '                    for idx in loaded_shard_id:\n'
                    '                        param.shard_weight_type[idx] = loaded_weight.item()'
                )
                new_subpatch = (
                    '                if isinstance(loaded_shard_id, tuple):\n'
                    '                    param.data[list(loaded_shard_id)].copy_(loaded_weight)\n'
                    '                    param.shard_weight_type[loaded_shard_id] = loaded_weight.item()\n'
                    '                    for idx in loaded_shard_id:\n'
                    '                        param.shard_weight_type[idx] = loaded_weight.item()'
                )
                if old_subpatch in content:
                    content = content.replace(old_subpatch, new_subpatch)
                    changed = True
                    print("[Patch] Updated linear.py to add tuple key mapping in shard_weight_type")
                else:
                    print("[Patch] linear.py patch already applied with tuple key mapping.")
            else:
                print("[Patch] WARNING: target_lin not matched in linear.py")

        if changed:
            with open(file_path_lin, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[Patch] Successfully wrote linear.py changes!")
    except Exception as e:
        print(f"[Patch] Error patching linear.py: {e}")
else:
    print(f"[Patch] linear.py not found at {file_path_lin}")

# 8. Patch vllm models/config.py to add Qwen3_5ForCausalLM and Qwen3_5MoeForCausalLM to MODELS_CONFIG_MAP
file_path_config = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/config.py'
if os.path.exists(file_path_config):
    try:
        with open(file_path_config, 'r', encoding='utf-8') as f:
            content = f.read()

        content = content.replace('\r\n', '\n')
        changed = False

        # Add class definition
        class_target = 'class Qwen3_5ForConditionalGenerationConfig(VerifyAndUpdateConfig):'
        class_replacement = (
            'class Qwen3_5CausalLMConfig(HybridAttentionMambaModelConfig):\n'
            '    @classmethod\n'
            '    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:\n'
            '        super().verify_and_update_config(vllm_config)\n'
            '        cache_config = vllm_config.cache_config\n'
            '        hf_text_config = vllm_config.model_config.hf_text_config\n'
            '        mamba_ssm_dtype = getattr(hf_text_config, "mamba_ssm_dtype", None)\n'
            '        if cache_config.mamba_ssm_cache_dtype == "auto":\n'
            '            if mamba_ssm_dtype is not None:\n'
            '                cache_config.mamba_ssm_cache_dtype = mamba_ssm_dtype\n\n\n'
            'class Qwen3_5ForConditionalGenerationConfig(VerifyAndUpdateConfig):'
        )

        if class_target in content and 'Qwen3_5CausalLMConfig' not in content:
            content = content.replace(class_target, class_replacement)
            changed = True
            print("[Patch] Added Qwen3_5CausalLMConfig to config.py")

        # Add registry in MODELS_CONFIG_MAP
        map_target = '"Qwen3_5ForConditionalGeneration": Qwen3_5ForConditionalGenerationConfig,'
        map_replacement = (
            '"Qwen3_5ForCausalLM": Qwen3_5CausalLMConfig,\n'
            '    "Qwen3_5MoeForCausalLM": Qwen3_5CausalLMConfig,\n'
            '    "Qwen3_5ForConditionalGeneration": Qwen3_5ForConditionalGenerationConfig,'
        )

        if map_target in content and '"Qwen3_5ForCausalLM"' not in content:
            content = content.replace(map_target, map_replacement)
            changed = True
            print("[Patch] Registered Qwen3_5ForCausalLM and Qwen3_5MoeForCausalLM in MODELS_CONFIG_MAP")

        if changed:
            with open(file_path_config, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[Patch] Successfully patched config.py!")
        else:
            print("[Patch] config.py patches already applied.")
    except Exception as e:
        print(f"[Patch] Error patching config.py: {e}")
else:
    print(f"[Patch] config.py not found at {file_path_config}")


# 9. Patch vllm/v1/core/kv_cache_utils.py to bypass MambaSpec layers in unify_kv_cache_spec_page_size
file_path_kv_utils = '/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py'
if os.path.exists(file_path_kv_utils):
    try:
        with open(file_path_kv_utils, 'r', encoding='utf-8') as f:
            content = f.read()

        changed_kv = False

        target_unify = '''def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Raise
    NotImplementedError if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec'''

        replacement_unify = '''def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Raise
    NotImplementedError if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    mamba_specs = {k: v for k, v in kv_cache_spec.items() if v.__class__.__name__ == "MambaSpec"}
    attn_specs = {k: v for k, v in kv_cache_spec.items() if v.__class__.__name__ != "MambaSpec"}
    if not attn_specs:
        return kv_cache_spec

    page_sizes = {layer.page_size_bytes for layer in attn_specs.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        unified_attn = attn_specs
    else:
        max_page_size = max(page_sizes)
        unified_attn = {}
        for layer_name, layer_spec in attn_specs.items():
            if layer_spec.page_size_bytes == max_page_size:
                unified_attn[layer_name] = layer_spec
            else:
                layer_page_size = layer_spec.page_size_bytes
                if max_page_size % layer_page_size != 0:
                    raise NotImplementedError(
                        "The page size of the layer is not divisible by the "
                        "maximum page size. Cannot unify by adjusting block_size."
                    )
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
                assert new_spec.page_size_bytes == max_page_size
                unified_attn[layer_name] = new_spec
    return {**unified_attn, **mamba_specs}'''

        content = content.replace('\r\n', '\n')
        if target_unify in content and 'mamba_specs' not in content:
            content = content.replace(target_unify, replacement_unify)
            changed_kv = True
            print("[Patch] Successfully patched kv_cache_utils.py for MambaSpec bypass!")

        # get_uniform_page_size patch
        target_uniform = '''def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """
    Get the page size of the KV cache.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()'''

        replacement_uniform = '''def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """
    Get the page size of the KV cache.
    """
    filtered_specs = [spec for spec in kv_cache_specs if spec.__class__.__name__ != "MambaSpec"]
    if not filtered_specs:
        filtered_specs = list(kv_cache_specs)
    page_sizes = {layer.page_size_bytes for layer in filtered_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()'''

        if target_uniform in content and 'filtered_specs' not in content:
            content = content.replace(target_uniform, replacement_uniform)
            changed_kv = True
            print("[Patch] Successfully patched kv_cache_utils.py for uniform page size MambaSpec bypass!")

        # get_kv_cache_config_from_groups patch
        target_config = '''        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )'''

        replacement_config = '''        mamba_groups = [g for g in kv_cache_groups if g.kv_cache_spec.__class__.__name__ == "MambaSpec"]
        attn_groups = [g for g in kv_cache_groups if g.kv_cache_spec.__class__.__name__ != "MambaSpec"]

        attn_group_size = max(len(group.layer_names) for group in attn_groups) if attn_groups else 0
        attn_page_size = 0
        if attn_groups:
            attn_page_size = get_uniform_page_size(
                [group.kv_cache_spec for group in attn_groups]
            )

        total_page_bytes = attn_page_size * attn_group_size
        for group in mamba_groups:
            total_page_bytes += group.kv_cache_spec.page_size_bytes * len(group.layer_names)

        assert total_page_bytes > 0, "total_page_bytes must be greater than 0"
        num_blocks = int(available_memory // total_page_bytes)
        num_blocks = max(num_blocks, 0)
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)

        kv_cache_tensors = []
        if attn_groups:
            for i in range(attn_group_size):
                shared_by = []
                for j in range(len(attn_groups)):
                    if i < len(attn_groups[j].layer_names):
                        shared_by.append(attn_groups[j].layer_names[i])
                kv_cache_tensors.append(
                    KVCacheTensor(size=attn_page_size * num_blocks, shared_by=shared_by)
                )

        for group in mamba_groups:
            mamba_ps = group.kv_cache_spec.page_size_bytes
            for layer_name in group.layer_names:
                kv_cache_tensors.append(
                    KVCacheTensor(size=mamba_ps * num_blocks, shared_by=[layer_name])
                )'''

        if target_config in content and 'mamba_groups' not in content:
            content = content.replace(target_config, replacement_config)
            changed_kv = True
            print("[Patch] Successfully patched kv_cache_utils.py for hybrid Mamba/Attention cache allocation!")

        if changed_kv:
            with open(file_path_kv_utils, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            print("[Patch] kv_cache_utils.py patches already applied or not matched.")
    except Exception as e:
        print(f"[Patch] Error patching kv_cache_utils.py: {e}")
else:
    print(f"[Patch] kv_cache_utils.py not found at {file_path_kv_utils}")


# 10. Patch vllm/model_executor/layers/quantization/gguf.py to add fallback logic for missing shard_weight_types
file_path_gguf = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/gguf.py'
if os.path.exists(file_path_gguf):
    try:
        with open(file_path_gguf, 'r', encoding='utf-8') as f:
            content = f.read()

        content = content.replace('\r\n', '\n')
        changed = False

        target_gguf = '''            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                result.append(
                    fused_mul_mat_gguf(
                        x, qweight[start:end, :offset].contiguous(), qweight_type
                    )
                )'''

        replacement_gguf = '''            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type.get(idx)
                if qweight_type is None:
                    shard_weight = qweight[start:end, :offset]
                    if shard_weight.dtype == torch.float32:
                        qweight_type = 0
                    elif shard_weight.dtype == torch.float16:
                        qweight_type = 1
                    elif shard_weight.dtype == torch.bfloat16:
                        qweight_type = 30
                    else:
                        qweight_type = 0
                result.append(
                    fused_mul_mat_gguf(
                        x, qweight[start:end, :offset].contiguous(), qweight_type
                    )
                )'''

        if target_gguf in content and 'shard_weight_type.get(idx)' not in content:
            content = content.replace(target_gguf, replacement_gguf)
            with open(file_path_gguf, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[Patch] Successfully patched gguf.py for missing shard weight type fallback!")
        else:
            print("[Patch] gguf.py patch already applied or not matched.")
    except Exception as e:
        print(f"[Patch] Error patching gguf.py: {e}")
else:
    print(f"[Patch] gguf.py not found at {file_path_gguf}")


# 11. Patch vllm/v1/worker/gpu_model_runner.py to bypass MambaSpec page size assertion check in _reshape_kv_cache_tensors
file_path_runner = '/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py'
if os.path.exists(file_path_runner):
    try:
        with open(file_path_runner, 'r', encoding='utf-8') as f:
            content = f.read()

        content = content.replace('\r\n', '\n')
        changed = False

        target_runner = '''                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):'''

        replacement_runner = '''                raw_tensor = kv_cache_raw_tensors[layer_name]
                if kv_cache_spec.__class__.__name__ == "MambaSpec":
                    num_blocks = self.kv_cache_config.num_blocks
                else:
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):'''

        if target_runner in content and 'kv_cache_spec.__class__.__name__ == "MambaSpec"' not in content:
            content = content.replace(target_runner, replacement_runner)
            with open(file_path_runner, 'w', encoding='utf-8') as f:
                f.write(content)
            changed = True
            print("[Patch] Successfully patched gpu_model_runner.py to bypass MambaSpec assertion!")
        else:
            print("[Patch] gpu_model_runner.py patch already applied or not matched.")
    except Exception as e:
        print(f"[Patch] Error patching gpu_model_runner.py: {e}")
else:
    print(f"[Patch] gpu_model_runner.py not found at {file_path_runner}")



