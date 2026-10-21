import os
from numpy import save
import torch
import argparse
from tqdm import tqdm
from safetensors.torch import save_file

from diffsynth.pipelines.wan_video import ModelConfig
from diffsynth.pipelines.wan_video_embeder import WanVideoEmbeder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_prompt(txt_path):
    with open(txt_path, 'r', encoding="utf-8") as f:
        content = f.read()
    return content

def save_prompt_embeddings(prompt_tensor, save_path):


    tensors = {
        "text_embedding": prompt_tensor
    }

    if save_path.endswith('.txt'):
        save_path = save_path.replace('.txt', '.safetensors')
    
    save_file(tensors, save_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_dir", type=str, default="data/mix", help="Directory containing text files with prompts.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the embedding files.")
    args = parser.parse_args()
    txt_dir = args.txt_dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = txt_dir

    preset_prompt = None

    pipe = WanVideoEmbeder.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
                    ModelConfig(
                        model_id="Wan-AI/Wan2.2-TI2V-5B", 
                        path="models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth", 
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"
                        ),
                ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            path="models/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl", 
            origin_file_pattern="google/umt5-xxl/"
            ),
    )


    prompt_paths = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    for prompt_path in tqdm(prompt_paths):
        full_path = os.path.join(txt_dir, prompt_path)
        prompt = read_prompt(full_path)

        if preset_prompt is not None:
            prompt = preset_prompt + prompt

        with torch.no_grad():
            prompt_embedding = pipe(prompt)


        prompt_embedding = prompt_embedding.cpu().detach()
        if len(prompt_embedding.shape) == 3:
            prompt_embedding = prompt_embedding.squeeze(0)
            
        save_path = os.path.join(output_dir, prompt_path.replace('.txt', '.safetensors'))
        save_prompt_embeddings(prompt_embedding, save_path)