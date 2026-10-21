from ast import parse
from email.mime import audio
from unittest import skip
from numpy import add, save
import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video_contlora import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.diffusion.runner import launch_training_task_with_tensorboard, launch_data_process_task
from diffsynth.diffusion.logger import TensorboardLogger
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        additional_model_configs: dict = {},
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Multi-condition training configs
        self.additonal_cond_num = additional_model_configs.get("additional_cond_num", None)
        self.no_clip = additional_model_configs.get("no_clip", False)
        self.drop_all_cond_prob = additional_model_configs.get("drop_all_cond_prob", None)
        self.drop_basecolor_cond_prob = additional_model_configs.get("drop_basecolor_cond_prob", None)
        self.drop_other_cond_prob = additional_model_configs.get("drop_other_cond_prob", None)
        self.inject_noise_strenghth = additional_model_configs.get("inject_noise_strenghth", None)
        self.replace_first_frame_prob = additional_model_configs.get("replace_first_frame_prob", None)

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = None
        audio_processor_config = None

        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config, additional_model_configs=additional_model_configs, redirect_common_files=False)
        
        if preset_lora_path is not None:
            self.pipe.load_lora_and_patch_embedding(self.pipe.dit, preset_lora_path)

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task, additonal_cond_num=self.additonal_cond_num,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train_2lr": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        total_params = sum(p.numel() for p in self.pipe.parameters())

        trainable_params = sum(p.numel() for p in self.pipe.parameters() if p.requires_grad)

        print(f"Trainable models: {trainable_models}")
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Trainable Ratio: {trainable_params / total_params:.4f}")

    

    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        task="sft", additonal_cond_num=None
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)

        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Preset LoRA
        if preset_lora_path is not None:
            if additonal_cond_num is None:
                pipe.load_lora(getattr(pipe, preset_lora_model), preset_lora_path)
            else:
                pipe.load_lora_and_patch_embedding(getattr(pipe, preset_lora_model), preset_lora_path)
        
        # FP8
        # FP8 relies on a model-specific memory management scheme.
        # It is delegated to the subclass.
        
        # Add LoRA to the base models
        if lora_base_model is not None and not task.endswith(":data_process"):
            if (not hasattr(pipe, lora_base_model)) or getattr(pipe, lora_base_model) is None:
                print(f"No {lora_base_model} models in the pipeline. We cannot patch LoRA on the model. If this occurs during the data processing stage, it is normal.")
                return
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)

        pipe.activate_module([] if trainable_models is None else trainable_models.split(","))


    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "additonal_cond_num": self.additonal_cond_num,
            "no_clip": self.no_clip,
            "drop_all_cond_prob": self.drop_all_cond_prob,
            "drop_basecolor_cond_prob": self.drop_basecolor_cond_prob,
            "drop_other_cond_prob": self.drop_other_cond_prob,
            "inject_noise_strenghth": self.inject_noise_strenghth,
            "replace_first_frame_prob": self.replace_first_frame_prob,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def add_mulcond_config(parser: argparse.ArgumentParser):
    parser.add_argument("--no_clip", action='store_true', help="Whether to use CLIP feature.")
    parser.add_argument("--second_learning_rate", type=float, default=None, help="Learning rate for the second parameter group.")
    parser.add_argument("--drop_all_cond_prob", type=float, default=None, help="drop all condition probability for multi-condition training.")
    parser.add_argument("--drop_basecolor_cond_prob", type=float, default=None, help="drop basecolor condition probability for multi-condition training.")
    parser.add_argument("--drop_other_cond_prob", type=float, default=None, help="drop basecolor condition probability for multi-condition training.")
    parser.add_argument("--inject_noise_strenghth", type=float, default=None, help="inject noise strength for multi-condition training.")
    parser.add_argument("--replace_first_frame_prob", type=float, default=None, help="replace first frame probability for multi-condition training.")
    return parser

def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser = add_mulcond_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            #"input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "prompt": UnifiedDataset.safetensor_prompt_operator(base_path=args.dataset_base_path),
            "type": lambda x: x, "name": lambda x: x
        }
    )

    extra_inputs = args.extra_inputs.split(",")
    extra_video_inputs = [x for x in extra_inputs if x.endswith("_video")]
    additional_model_configs = {
        "additional_cond_num": len(extra_video_inputs),
        "no_clip": args.no_clip,
        "drop_all_cond_prob": args.drop_all_cond_prob,
        "drop_basecolor_cond_prob": args.drop_basecolor_cond_prob,
        "drop_other_cond_prob": args.drop_other_cond_prob,
        "inject_noise_strenghth": args.inject_noise_strenghth,
        "replace_first_frame_prob": args.replace_first_frame_prob,
    }

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        additional_model_configs=additional_model_configs,
    )
    model_logger = TensorboardLogger(
        args.output_path,
    )
    
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task_with_tensorboard,
        "sft:train": launch_training_task_with_tensorboard,
        "direct_distill": launch_training_task_with_tensorboard,
        "direct_distill:train": launch_training_task_with_tensorboard,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
