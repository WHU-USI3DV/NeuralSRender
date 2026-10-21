/opt/conda/bin/python examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B-cond-lora.py \
    --dataset_base_path "<path to data>" \
    --metadata_path "<path to metadata>" \
    --base_ckpt_path "<path to synthesis lora ckpt>" \
    --addition_ckpt_path "<path to real lora ckpt>" \
    --output_path "outputs/nsrender/test" \
    --enable_offload \
    --num_inference_steps 30 \
    --num_frames 81 \
    --max_data_index 100000 \

# Infer Long Video
python examples/wanvideo/model_training/validate_adapter/Wan2.2-TI2V-5B-cond-lora-arg-vr.py \
    --dataset_base_path "<path to data>" \
    --metadata_path "<path to metadata>" \
    --base_ckpt_path "<path to synthesis lora ckpt>" \
    --addition_ckpt_path "<path to real lora ckpt>" \
    --output_path "outputs/nsrender/long_test" \
    --enable_offload \
    --num_inference_steps 30 \
    --max_data_index 100000 \
    --num_frames 4017 \
    --sliding_window_size 30 --sliding_window_stride 15 \