from curses import meta
from turtle import width
import torch
from safetensors.torch import load_file
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video_contlora_infer import ModelConfig
from modelscope import dataset_snapshot_download
import torch, os, argparse, warnings
import accelerate # 引入 accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadPromptSafetensor, LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video_contlora_infer import WanVideoPipeline
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    set_seed(args.seed + accelerator.process_index) 

    if accelerator.is_main_process:
        print(f"Starting inference with {accelerator.num_processes} processes.")

    dataset_base_path = args.dataset_base_path
    metadata_path = args.metadata_path
    negative_prompt_path = args.negative_prompt_path
    base_ckpt_path = args.base_ckpt_path
    addition_ckpt_path = args.addition_ckpt_path
    output_path = args.output_path
    data_index = args.data_index
    num_frames = args.num_frames
    height = args.height
    width = args.width
    sliding_window_size = args.sliding_window_size
    sliding_window_stride = args.sliding_window_stride
    cfg_scale = args.cfg_scale
    tile_size = tuple(args.tile_size)
    tile_stride = tuple(args.tile_stride)
    enable_offload = args.enable_offload
    num_inference_steps = args.num_inference_steps

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    dataset = UnifiedDataset(
        base_path=dataset_base_path,
        metadata_path=metadata_path,
        data_file_keys="video,prompt,basecolor_video,depth_video,normal_video,metallic_video,roughness_video".split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=dataset_base_path,
            max_pixels=1280*720,
            height=height,
            width=width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(dataset_base_path) >> LoadVideo(num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            #"input_audio": ToAbsolutePath(dataset_base_path) >> LoadAudio(sr=16000),
            "prompt": UnifiedDataset.default_prompt_operator(base_path=dataset_base_path),
            "type": lambda x: x, "name": lambda x: x
        }
    )

    if enable_offload:
        vram_config = {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": device,
            "computation_dtype": torch.bfloat16,
            "computation_device": device, 
        }
    else:
        vram_config = {}

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device, 
        additional_model_configs = {"additional_cond_num": 5, "no_clip": True,},
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B", 
                path=[
                    "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003-bf16.safetensors",
                    "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003-bf16.safetensors",
                    "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003-bf16.safetensors",
                ],
                **vram_config,
            ),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", path="models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth", origin_file_pattern="Wan2.2_VAE.pth", **vram_config),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", path="models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ],
    )

    pipe.load_lora_and_patch_embedding(pipe.dit, base_ckpt_path, alpha=1)
    if addition_ckpt_path is not None:
        pipe.load_lora(pipe.dit, addition_ckpt_path, alpha=1)

    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, 变形的车辆，弯曲的栏杆"

    if args.max_data_index is None:
        full_indices = [data_index]
    else:
        max_idx = min(args.max_data_index, len(dataset))
        full_indices = list(range(data_index, max_idx))

    local_indices = [i for i in full_indices if i % accelerator.num_processes == accelerator.process_index]
    
    print(f"[Process {accelerator.process_index}] Assigned tasks: {len(local_indices)} videos")

    for idx in local_indices:
        data = dataset.__getitem__(idx)
        video_name = data['name']
        output_video_path = f"{output_path}/{video_name}_{height}p_{num_frames}.mp4"

        if os.path.exists(output_video_path):
            print(f"[Process {accelerator.process_index}] Skip existing video: {output_video_path}")
            continue

        print(f"[Process {accelerator.process_index}] Processing: {video_name}")

        try:
            current_seed = args.seed + idx 
            
            video = pipe(
                prompt=data["prompt"], 
                negative_prompt=negative_prompt,
                basecolor_video=data["basecolor_video"], 
                depth_video=data["depth_video"], 
                normal_video=data["normal_video"], 
                metallic_video=data["metallic_video"], 
                roughness_video=data["roughness_video"],
                height=height, 
                width=width, 
                num_frames=num_frames,
                seed=current_seed, 
                tiled=True, 
                cfg_scale=cfg_scale,
                sliding_window_size=sliding_window_size, 
                sliding_window_stride=sliding_window_stride,
                tile_size=tile_size,
                tile_stride=tile_stride,
                num_inference_steps=num_inference_steps,
            )

            save_video(video, output_video_path, fps=10, quality=5)
            print(f"[Process {accelerator.process_index}] Saved: {output_video_path}")
            
        except Exception as e:
            print(f"[Process {accelerator.process_index}] Error processing {video_name}: {e}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("All processes finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B Video Generation with Accelerate")

    # 添加命令行参数
    parser.add_argument("--dataset_base_path", type=str, default="data/long_test_704p", help="Base path for the dataset")
    parser.add_argument("--metadata_path", type=str, default="data/long_test_704p/citydataset_metadata_safetensors.csv", help="Path to the metadata file")
    parser.add_argument("--negative_prompt_path", type=str, default="data/long_test/negative_prompt.safetensors", help="Path to the negative prompt file")
    parser.add_argument("--base_ckpt_path", type=str, default="models/train/Wan2.2-TI2V-5B/contlora/nuc/base/step-10000.safetensors", help="Path to the checkpoint file")
    parser.add_argument("--addition_ckpt_path", type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument("--output_path", type=str, default="outputs/Wan2.2-TI2V-5B/control_lora/nuc/base/10k_81f", help="Path to save the output")
    parser.add_argument("--data_index", type=int, default=0, help="Index of the data to process")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the video")
    parser.add_argument("--height", type=int, default=704, help="Height of the video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the video")
    parser.add_argument("--max_data_index", type=int, default=None, help="Maximum data index to process")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Sliding window size for processing")
    parser.add_argument("--sliding_window_size", type=int, default=None, help="Sliding window size for processing")
    parser.add_argument("--sliding_window_stride", type=int, default=None, help="Sliding window stride for processing")
    parser.add_argument("--tile_size", type=int, nargs=2, default=[30, 52], help="Tile size for processing (height, width)")
    parser.add_argument("--tile_stride", type=int, nargs=2, default=[15, 26], help="Tile stride for processing (height, width)")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale for processing")
    parser.add_argument("--enable_offload", action="store_true", help="Enable offload for processing")
    parser.add_argument("--seed", type=int, default=1, help="Base seed for generation") # 添加 seed 参数
    

    args = parser.parse_args()

    main(args)