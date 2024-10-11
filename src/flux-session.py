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
sys.path.append("src/utils")
from local_os import does_file_exist
from local_logs import log_session, log_debug, log_inference, log_error, log_warning
from local_inputs import input_instruction


def load_envs():
  load_dotenv()
  log_debug("Setting variables...")
  model_variant = sys.argv[1]
  log_debug(f"Model version: {model_variant}")
  log_debug("\nLoading environment variables...")
  negative_prompt = os.getenv("NEGATIVE_PROMPT")
  num_inference_steps = int(os.getenv("NUM_INFERENCE_STEPS"))
  guidance_scale = float(os.getenv("GUIDANCE_SCALE"))
  width = int(os.getenv("WIDTH") or 1024)
  height = int(os.getenv("HEIGHT") or 1024)
  num_images_per_prompt = int(os.getenv("NUM_IMAGES_PER_PROMPT"))
  debug_session = os.getenv("DEBUG_SESSION").lower() == "true"

  log_debug("\nDebugging environment...")
  log_debug(f"Torch version: {torch.__version__}")
  log_debug(f"CUDA available: {torch.cuda.is_available()}")
  log_debug(f"CUDA version: {torch.version.cuda}")
  log_debug(f"CUDNN version: {torch.backends.cudnn.version()}")
  log_debug(f"CUDA device count: {torch.cuda.device_count()}")
  log_debug(f"CUDA device name: {torch.cuda.get_device_name()}")

  return model_variant, negative_prompt, num_inference_steps, guidance_scale, width, height, num_images_per_prompt, debug_session


def login_huggingface():
  log_debug("\nLogging into huggingface...")
  token = os.getenv("HUGGING_FACE_API_KEY")
  if token is None:
      log_error("HUGGING_FACE_API_KEY is not set")
      sys.exit(1)

  login(token=token, add_to_git_credential=True)


def flush():
    log_inference("\nFlushing memory...")
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def inference(filename, prompt, text_encoder, pipeline, width, height, guidance_scale, num_inference_steps, num_images_per_prompt):
    log_inference("\nEncoding prompt...")
    text_encoder.to("cuda")
    encodingStart = time.time()
    (
        prompt_embeds,
        pooled_prompt_embeds,
        _,
    ) = text_encoder.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=512)
    text_encoder.to("cpu")

    # Set manual seed for reproducibility
    # torch.manual_seed(0)

    flush()
    log_inference(f"Prompt encoding time: {time.time() - encodingStart}")

    log_inference("\nGenerating image...")
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
    images = output.images
    for i, image in enumerate(images):
      if len(images) == 1:
        image_name = filename
      else:
        image_name = f"{filename}({i})"

      log_inference(f"\nSaving image '{filename}'...")

      file_path = f"dist/{image_name}.png"
      new_image_name = image_name
      image_name_taken = does_file_exist(file_path)
      while image_name_taken:
        log_warning(f"File '{file_path}' already exists.")
        new_image_name = f"{new_image_name}-(copy)"
        new_file_path = f"dist/{new_image_name}.png"
        new_image_name_taken = does_file_exist(new_file_path)
        if new_image_name_taken:
          log_warning(f"File '{new_file_path}' already exists.")
          continue
        else:
          image_name_taken = False
          file_path = new_file_path

      image.save(file_path)


def setup_model(model_variant):
  log_inference("\nLoading model...")
  t5_encoder = T5EncoderModel.from_pretrained(
      f"black-forest-labs/FLUX.1-{model_variant}",
      subfolder="text_encoder_2",
      torch_dtype=torch.bfloat16
  )
  text_encoder = diffusers.DiffusionPipeline.from_pretrained(
      f"black-forest-labs/FLUX.1-{model_variant}",
      text_encoder_2=t5_encoder,
      transformer=None,
      vae=None,
      # revision="refs/pr/7",
  )
  pipeline = diffusers.DiffusionPipeline.from_pretrained(
      f"black-forest-labs/FLUX.1-{model_variant}",
      torch_dtype=torch.bfloat16,
      # revision="refs/pr/1",
      text_encoder_2=None,
      text_encoder=None,
  )
  pipeline.enable_model_cpu_offload()

  quantizeStart = time.time()
  log_inference("\nQuantizing model...")
  quantize(pipeline.transformer, weights=qfloat8)
  freeze(pipeline.transformer)
  log_inference(f"Quantizing time: {time.time() - quantizeStart}")

  return text_encoder, pipeline


def loop_session(debug_session, text_encoder, pipeline, width, height, guidance_scale, num_inference_steps, num_images_per_prompt, negative_prompt):
  last_file_name = ""
  last_prompt = ""
  while True:
      first_attempt = last_file_name == "" and last_prompt == "";
      if first_attempt:
        log_session("\nCreating first image...")
      else:
        log_session("\nCreating another image...")

      if debug_session:
          log_debug(f"\nCurrent settings:")
          log_debug(f"|--Negative prompt: {negative_prompt}")
          log_debug(f"|--Number of inference steps: {num_inference_steps}")
          log_debug(f"|--Guidance scale: {guidance_scale}")
          log_debug(f"|--Width: {width}")
          log_debug(f"|--Height: {height}")
          log_debug(f"|--Number of images per prompt: {num_images_per_prompt}\n")
      if first_attempt:
          user_command = input_instruction("Enter command:\n - 'a' to continue normally by inserting filename and prompt\n - 'b' to continue with advanced settings\nor anything else to exit the session: ")
      else:
          user_command = input_instruction("Enter command:\n - 'a' to continue normally by inserting filename and prompt\n - 'b' to continue with advanced configuration\n - 'c' to continue with last configuration\nor anything else to exit the session: ")

      if user_command.lower() == "a":
          last_file_name = input_instruction("\nEnter the filename: ")
          last_prompt = input_instruction("Enter the prompt: ")
      elif user_command.lower() == "b":
          last_file_name = input_instruction("\nEnter the filename: ")
          last_prompt = input_instruction("Enter the prompt: ")
          num_inference_steps = int(input_instruction("Enter the number of inference steps: "))
          guidance_scale = float(input_instruction("Enter the guidance scale: "))
          width = int(input_instruction("Enter the width: "))
          height = int(input_instruction("Enter the height: "))
          num_images_per_prompt = int(input_instruction("Enter the number of images per prompt: "))
      elif user_command.lower() == "c":
          pass
      else:
        break

      inference(
         prompt=last_prompt,
         filename=last_file_name,
         text_encoder=text_encoder,
         pipeline=pipeline,
         width=width,
         height=height,
         guidance_scale=guidance_scale,
         num_inference_steps=num_inference_steps,
         num_images_per_prompt=num_images_per_prompt
      )


def main():
  start = time.time()

  if len(sys.argv) < 2:
    log_session(f"Usage: python flux-session.py <model>(dev / schnell)")
    sys.exit(1)

  model_variant, negative_prompt, num_inference_steps, guidance_scale, width, height, num_images_per_prompt, debug_session = load_envs()
  login_huggingface()
  text_encoder, pipeline = setup_model(model_variant=model_variant)
  loop_session(
     debug_session=debug_session,
     text_encoder=text_encoder,
     pipeline=pipeline,
     width=width,
     height=height,
     guidance_scale=guidance_scale,
     num_inference_steps=num_inference_steps,
     num_images_per_prompt=num_images_per_prompt,
     negative_prompt=negative_prompt
  )

  log_session(f"\nImage generation time: {time.time() - start}.")
  log_session("\nSession ended. Exiting...")


main()
