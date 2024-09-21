# This script does need torch, diffusers, accelerate, protobuf, sentencepiece to be installed. e.g. 'conda install diffusers'
# It also needs optimum-quanto, flash-attn, dotenvto be installed. e.g. 'pip install optimum-quanto'
# See https://huggingface.co/docs/diffusers/v0.9.0/en/api/pipelines/stable_diffusion
# See also https://huggingface.co/blog/quanto-diffusers
from transformers import T5EncoderModel
import time
import gc
import sys
import torch
import diffusers
from huggingface_hub import login
from optimum.quanto import freeze, qfloat8, quantize
import os
from dotenv import load_dotenv
load_dotenv()


start = time.time()

if len(sys.argv) < 2:
  print(f"Usage: python flux-session.py <model>(dev / schnell)")
  sys.exit(1)

print("Setting variables...")
model = sys.argv[1]
print(f"Model verison: {model}")
print("\nLoading environment variables...")
negative_prompt = os.getenv("NEGATIVE_PROMPT")
print(f"Negative prompt: {negative_prompt}")
num_inference_steps = int(os.getenv("NUM_INFERENCE_STEPS"))
print(f"Number of inference steps: {num_inference_steps}")
guidance_scale = float(os.getenv("GUIDANCE_SCALE"))
print(f"Guidance scale: {guidance_scale}")
width = int(os.getenv("WIDTH") or 1024)
print(f"Width: {width}")
height = int(os.getenv("HEIGHT") or 1024)
print(f"Height: {height}")
num_images_per_prompt = int(os.getenv("NUM_IMAGES_PER_PROMPT"))
print(f"Number of images per prompt: {num_images_per_prompt}")


print("\nDebugging environment...")
print("Torch version: ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())
print("CUDA version: ", torch.version.cuda)
print("CUDNN version: ", torch.backends.cudnn.version())
print("CUDA device count: ", torch.cuda.device_count())
print("CUDA device name: ", torch.cuda.get_device_name())

print("\nLogging into huggingface...")
token = os.getenv("HUGGING_FACE_API_KEY")
if token is None:
    print("HUGGING_FACE_API_KEY is not set")
    sys.exit(1)

login(token=token, add_to_git_credential=True)

def flush():
    print("\nFlushing memory...")
    gc.collect()
    torch.cuda.empty_cache()

print("\nLoading model...")
t5_encoder = T5EncoderModel.from_pretrained(
    f"black-forest-labs/FLUX.1-{model}", subfolder="text_encoder_2", torch_dtype=torch.bfloat16
)
text_encoder = diffusers.DiffusionPipeline.from_pretrained(
    f"black-forest-labs/FLUX.1-{model}",
    text_encoder_2=t5_encoder,
    transformer=None,
    vae=None,
    # revision="refs/pr/7",
)
pipeline = diffusers.DiffusionPipeline.from_pretrained(
    f"black-forest-labs/FLUX.1-{model}",
    torch_dtype=torch.bfloat16,
    # revision="refs/pr/1",
    text_encoder_2=None,
    text_encoder=None,
)
pipeline.enable_model_cpu_offload()

quantizeStart = time.time()
print("\nQuantizing model...")
quantize(pipeline.transformer, weights=qfloat8)
freeze(pipeline.transformer)
print(f"Quantizing time: {time.time() - quantizeStart}")

@torch.inference_mode()
def inference(text_encoder, pipeline, prompt, num_inference_steps, guidance_scale, width, height, num_images_per_prompt, filename):
    print("\nEncoding prompt...")
    text_encoder.to("cuda")
    encodingStart = time.time()
    (
        prompt_embeds,
        pooled_prompt_embeds,
        _,
    ) = text_encoder.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=256)
    text_encoder.to("cpu")
    flush()
    print(f"Prompt encoding time: {time.time() - encodingStart}")

    print("\nGenerating image...")
    output = pipeline(
        prompt_embeds=prompt_embeds.bfloat16(),
        pooled_prompt_embeds=pooled_prompt_embeds.bfloat16(),
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt
    )
    flush()
    image = output.images[0]
    print("\nSaving image...")
    image.save(f"dist/{filename}.png")

while True:
    print("\nCreating an image...")
    interactiveFilename = input("Enter the filename or type 'done' to exit: ")
    if interactiveFilename.lower() == "done":
        break

    interactivePrompt = input("Enter the prompt or type 'done' to exit: ")
    if interactivePrompt.lower() == "done":
        break

    inference(text_encoder=text_encoder, pipeline=pipeline, prompt=interactivePrompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=width, height=height, num_images_per_prompt=num_images_per_prompt, filename=interactiveFilename)

print(f"Image generation time: {time.time() - start}. Exiting the program.")
