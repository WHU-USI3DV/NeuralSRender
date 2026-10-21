import torch, warnings


class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype
    
    
    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_up." in key:
                lora_A_key = "lora_down"
                lora_B_key = "lora_up"
            else:
                lora_A_key = "lora_A"
                lora_B_key = "lora_B"
            if lora_B_key not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index(lora_B_key) + 2:
                keys.pop(keys.index(lora_B_key) + 1)
            keys.pop(keys.index(lora_B_key))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            # Alpha: Deprecated but retained for compatibility.
            key_alpha = key.replace(lora_B_key + ".weight", "alpha").replace(lora_B_key + ".default.weight", "alpha")
            if key_alpha == key or key_alpha not in lora_state_dict:
                key_alpha = None
            lora_name_dict[target_name] = (key, key.replace(lora_B_key, lora_A_key), key_alpha)
        return lora_name_dict
    
    
    def convert_state_dict(self, state_dict, suffix=".weight"):
        name_dict = self.get_name_dict(state_dict)
        state_dict_ = {}
        for name in name_dict:
            weight_up = state_dict[name_dict[name][0]]
            weight_down = state_dict[name_dict[name][1]]
            if name_dict[name][2] is not None:
                warnings.warn("Alpha detected in the LoRA file. This may be a LoRA model not trained by DiffSynth-Studio. To ensure compatibility, the LoRA weights will be converted to weight * alpha / rank.")
                alpha = state_dict[name_dict[name][2]] / weight_down.shape[0]
                weight_down = weight_down * alpha
            state_dict_[name + f".lora_B{suffix}"] = weight_up
            state_dict_[name + f".lora_A{suffix}"] = weight_down
        return state_dict_


    def fuse_lora_to_base_model(self, model: torch.nn.Module, state_dict, alpha=1.0):
        updated_num = 0
        state_dict = self.convert_state_dict(state_dict)
        lora_layer_names = set([i.replace(".lora_B.weight", "") for i in state_dict if i.endswith(".lora_B.weight")])
        for name, module in model.named_modules():
            if name in lora_layer_names:
                weight_up = state_dict[name + ".lora_B.weight"].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict[name + ".lora_A.weight"].to(device=self.device, dtype=self.torch_dtype)
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)
                state_dict_base = module.state_dict()
                state_dict_base["weight"] = state_dict_base["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict_base)
                updated_num += 1
        print(f"{updated_num} tensors are fused by LoRA. Fused LoRA layers cannot be cleared by `pipe.clear_lora()`.")



class LoRAWithPatchEmbeddingLoader(GeneralLoRALoader):
    def convert_state_dict(self, state_dict, suffix=".weight"):
        new_state_dict = super().convert_state_dict(state_dict, suffix)
        
        keys_to_preserve = ['patch_embedding.weight', 'patch_embedding.bias']
        for key in keys_to_preserve:
            if key in state_dict:
                new_state_dict[key] = state_dict[key]
        
        return new_state_dict

    def update_patch_embedding(self, model: torch.nn.Module, state_dict):

        patch_embedding_updated = False
        try:
            model_modules = dict(model.named_modules())
            module = model_modules.get('patch_embedding')
            
            if module is None:
                return

            with torch.no_grad():
                if 'patch_embedding.weight' in state_dict:
                    patch_weight_lora = state_dict['patch_embedding.weight'].to(self.device, self.torch_dtype)
                    current_weight = module.weight.data
                    lora_in_channels = patch_weight_lora.shape[1]
                    
                    if current_weight.shape[1] < lora_in_channels:
                        print(f"Warning: Model's patch_embedding in_channels ({current_weight.shape[1]}) is smaller than LoRA's ({lora_in_channels}). Cannot load.")
                    else:
                        print(f"Updating the first {lora_in_channels} input channels of patch_embedding.weight.")
                        current_weight[:, :lora_in_channels, :, :, :].copy_(patch_weight_lora)
                        patch_embedding_updated = True

                if 'patch_embedding.bias' in state_dict and module.bias is not None:
                    patch_bias_lora = state_dict['patch_embedding.bias'].to(self.device, self.torch_dtype)
                    module.bias.data.copy_(patch_bias_lora)
                    patch_embedding_updated = True

            if patch_embedding_updated:
                print("Tensors 'patch_embedding.weight' and/or 'patch_embedding.bias' were updated partially.")

        except Exception as e:
            print(f"An error occurred while loading patch_embedding weights: {e}")

    def fuse_lora_to_base_model(self, model: torch.nn.Module, state_dict, alpha=1.0):
        super().fuse_lora_to_base_model(model, state_dict, alpha)
        
        self.update_patch_embedding(model, state_dict)