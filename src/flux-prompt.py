# This script does need torch, diffusers, accelerate, protobuf, sentencepiece to be installed. e.g. 'conda install diffusers'
# It also needs optimum-quanto, flash-attn, dotenvto be installed. e.g. 'pip install optimum-quanto'
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
  print(f"Usage: python flux-prompt.py <model>(dev / schnell) <filename> <prompt>")
  sys.exit(1)


print("Setting variables...")
model = sys.argv[1]
print(f"Model verison: {model}")
filename = sys.argv[2]
print(f"Filename: {filename}")
default_prompt = "A bear in a forest."
prompt = sys.argv[3] if len(sys.argv) > 2 else default_prompt
print(f"Prompt: {prompt}")

print("\nDebugging environment...")
print("Torch version: ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())
print("CUDA version: ", torch.version.cuda)
print("CUDNN version: ", torch.backends.cudnn.version())
print("CUDA device count: ", torch.cuda.device_count())
print("CUDA device name: ", torch.cuda.get_device_name())

print("\nLogging into huggingface...")
# load token from environment variable HUGGING_FACE_API_KEY
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
    "black-forest-labs/FLUX.1-schnell",
    text_encoder_2=t5_encoder,
    transformer=None,
    vae=None,
    revision="refs/pr/7",
)
pipeline = diffusers.DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/1",
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
def inference(text_encoder, pipeline, prompt, num_inference_steps=4, guidance_scale=3.5, width=1024, height=1024):
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
        num_inference_steps=num_inference_steps
    )
    flush()
    image = output.images[0]
    return image

image = inference(text_encoder=text_encoder, pipeline=pipeline, prompt=prompt)
print("\nSaving image...")
image.save(f"images/{filename}.png")
print(f"Image generation time: {time.time() - start}")
