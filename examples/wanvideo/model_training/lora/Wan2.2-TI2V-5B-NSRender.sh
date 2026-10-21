
output_synthesis_path="./models/train/Wan2.2-TI2V-5B/contlora/city_synthesis/"

# 确保 output_path 目录存在
mkdir -p "$output_synthesis_path"

# 将当前脚本文件复制到 output_path
cp "$0" "$output_synthesis_path"

/opt/conda/bin/accelerate launch examples/wanvideo/model_training/train_cond_lora.py \
  --dataset_base_path "<path to synthesis data>" \
  --dataset_metadata_path "<path to synthesis metadata>" \
  --data_file_keys "video,prompt,basecolor_video,depth_video,normal_video,metallic_video,roughness_video" \
  --height 480 \
  --width 832 \
  --dataset_repeat 4 \
  --dataset_num_workers 8 \
  --model_paths '[
      [
        "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003-bf16.safetensors",
        "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003-bf16.safetensors",
        "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003-bf16.safetensors"
      ],
      "models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    ]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --save_steps 1000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$output_synthesis_path" \
  --trainable_models "dit.patch_embedding" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "basecolor_video,depth_video,normal_video,metallic_video,roughness_video,name,type" \
  --no_clip \
  --drop_basecolor_cond_prob 0.3 \
  --drop_other_cond_prob 0.2 \
  --drop_all_cond_prob 0.1 \



output_real_path="./models/train/Wan2.2-TI2V-5B/contlora/dv_real/"

# 确保 output_path 目录存在
mkdir -p "$output_real_path"

# 将当前脚本文件复制到 output_path
cp "$0" "$output_real_path"

/opt/conda/bin/accelerate launch examples/wanvideo/model_training/train_cond_lora.py \
  --dataset_base_path "<path to real data>" \
  --dataset_metadata_path "<path to real metadata>" \
  --data_file_keys "video,prompt,basecolor_video,depth_video,normal_video,metallic_video,roughness_video" \
  --height 480 \
  --width 832 \
  --dataset_repeat 4 \
  --dataset_num_workers 8 \
  --model_paths '[
      [
        "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003-bf16.safetensors",
        "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003-bf16.safetensors",
        "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003-bf16.safetensors"
      ],
      "models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    ]' \
  --learning_rate 1e-4 \
  --num_epochs 4 \
  --save_steps 1000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$output_real_path" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --preset_lora_model "dit" \
  --preset_lora_path "$output_synthesis_path/step-last.safetensors" \
  --lora_rank 32 \
  --extra_inputs "basecolor_video,depth_video,normal_video,metallic_video,roughness_video,name,type" \
  --no_clip \
  --drop_basecolor_cond_prob 0.6 \
  --drop_other_cond_prob 0.4 \
  --drop_all_cond_prob 0.2 