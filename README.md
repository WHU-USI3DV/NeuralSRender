# NeuralSRender: G-Buffer-Guided Video Diffusion for Sim-to-Real Rendering

This is the official PyTorch implementation of the following publication:

> **NeuralSRender: G-Buffer Guided Video Diffusion for Sim-to-Real Rendering**
> Yipeng Lu, Youqi Liao, Zhenyu Wang, Yuning Peng, Zhen Dong
> *arXiv preprint, 2026*
> [**Arxiv**](https://arxiv.org/abs/2610.29200) | [**Code**](https://github.com/WHU-USI3DV/NeuralSRender) | [**Model Weights**](https://huggingface.co/stardustlu/NeuralSRender)


![NeuralSRender teaser](assets/tesear.jpg)

## Overview

**TL;DR:** NeuralSRender uses geometry and material information from G-buffers to guide video diffusion, improving the realism of simulated driving videos while preserving scene geometry.

## Abstract

The scarcity of high-quality annotated visual data remains a critical bottleneck for autonomous-driving perception. Although simulators can generate training data at scale, the limited realism of synthetic videos often reduces their effectiveness in real-world applications.
To address this challenge, we introduce **NeuralSRender**, a sim-to-real video translation framework that enhances the photorealism of synthetic videos while maintaining geometric consistency with the source scenes. Built on a pretrained video diffusion transformer (DiT), NeuralSRender employs a lightweight projector to incorporate geometric and material priors from G-buffers, together with low-rank adaptation (LoRA) for parameter-efficient fine-tuning.
Training proceeds in two stages. First, the model learns G-buffer conditioning from synthetic videos with accurate G-buffer annotations. Second, it adapts to real-world appearance using real videos with estimated G-buffers. During the second stage, an additional LoRA module is trained while the projector and first-stage LoRA remain frozen, reducing interference from G-buffer estimation noise.
NeuralSRender introduces only **1.45% additional parameters** and can be trained on GPUs with **24 GB of memory**. We also introduce **GenReal**, a large-scale driving video dataset containing synthetic and real-world videos with corresponding G-buffers. Experiments show improved photorealism while maintaining competitive geometric consistency. When perception models are trained on our translated videos instead of the original simulator renders, detection AP50 on nuImages increases from **35.1 to 40.6**, and segmentation mIoU on Cityscapes improves from **32.4 to 40.8**.

## Demo and Results



https://github.com/user-attachments/assets/6204f1fa-d085-4dce-a01b-22572cb91b58



https://github.com/user-attachments/assets/c07903ac-d929-41d8-a5bb-116c8f1d0cce



https://github.com/user-attachments/assets/9a071682-dce5-417f-8f4c-37c8dff57ce6



## Model Checkpoints

The pretrained NeuralSRender model weights are available on Hugging Face.

You can download the checkpoints using the Hugging Face CLI:

```shell
hf download stardustlu/NeuralSRender \
    --local-dir ./models/NeuralSRender
```

## Quick Start

### Installation

Install NeuralSRender in editable mode:

```bash
pip install -e .
```

We recommend using Docker for a reproducible environment. Our development environment uses the following base image:

```text
pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
```

### Inference

An example inference script is available at:

```text
examples/wanvideo/model_training/validate_lora/infer.sh
```

#### Short-video inference

To generate an 81-frame video:

```bash
python examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B-cond-lora.py \
    --dataset_base_path "<path-to-data>" \
    --metadata_path "<path-to-metadata>" \
    --base_ckpt_path "<path-to-synthetic-stage-lora>" \
    --addition_ckpt_path "<path-to-real-stage-lora>" \
    --output_path "outputs/nsrender/test" \
    --enable_offload \
    --num_inference_steps 30 \
    --num_frames 81 \
    --max_data_index 100000
```

#### Long-video inference

For longer sequences, NeuralSRender supports **sliding-window inference**:

```bash
python examples/wanvideo/model_training/validate_adapter/Wan2.2-TI2V-5B-cond-lora-arg-vr.py \
    --dataset_base_path "<path-to-data>" \
    --metadata_path "<path-to-metadata>" \
    --base_ckpt_path "<path-to-synthetic-stage-lora>" \
    --addition_ckpt_path "<path-to-real-stage-lora>" \
    --output_path "outputs/nsrender/long_test" \
    --enable_offload \
    --num_inference_steps 30 \
    --max_data_index 100000 \
    --num_frames 4017 \
    --sliding_window_size 20 \
    --sliding_window_stride 10
```

We have tested video generation with up to **4,017 frames**. Longer sequences may encounter a limitation related to `dit.freqs` in the current implementation.

### Training

An example training script is available at:

```text
examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-NSRender.sh
```

In our training configuration, peak GPU memory usage is approximately **17 GB for single-GPU training** and **24 GB for multi-GPU training**. This makes training feasible on GPUs such as the NVIDIA RTX 4090. 

### Data Preparation

We use video clips with a spatial resolution of **832 × 480** (width × height) and **81 frames** for training. Each sample contains an RGB video and G-buffer videos for the following attributes:

- **Base color** (`basecolor`)
- **Depth** (`depth`)
- **Surface normals** (`normal`)
- **Metallic** (`metallic`)
- **Roughness** (`roughness`)

The dataset metadata should be provided as a CSV file with the following columns:

```csv
video,prompt,basecolor_video,depth_video,normal_video,metallic_video,roughness_video,name,type
```

**Additional notes:**

- **Optional G-buffer inputs:** Not all G-buffer attributes are required. Unused G-buffer inputs can be filled with zeros.
- **Frame count:** The number of frames must satisfy `4n + 1` for an integer `n` (e.g., 81 or 4017).
- **Prompt embeddings:** We recommend precomputing text prompt embeddings and storing them in `.safetensors` format to reduce GPU memory usage during training.
- **Long-video quality:** In our experiments, training with longer clips sometimes resulted in less vivid, visually flatter generations. We therefore recommend sliding-window inference for long videos.

## GenReal Dataset



https://github.com/user-attachments/assets/d9f70d09-7cbb-4396-bba6-f4a0990b0f0e




## Release Status

- [x] Training code
- [x] Inference code
- [ ] GenReal dataset

## Citation

If you find NeuralSRender useful for your research, please consider citing our paper:

```bibtex
@article{lu2026neuralsrender,
  title={NeuralSRender: G-Buffer Guided Video Diffusion for Sim-to-Real Rendering},
  author={Lu, Yipeng and Liao, Youqi and Wang, Zhenyu and Peng, Yuning and Dong, Zhen},
  journal={arXiv preprint arXiv:2610.29200},
  year={2026},
  url={https://arxiv.org/abs/2610.29200}
}
```

## Acknowledgements

This repository builds upon [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/tree/main). We thank its contributors for making their work publicly available.
