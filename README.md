# blackforest-flux

## Pre-requisites
- env file with the following content:
```env
HUGGING_FACE_API_KEY=<your-hugging-face-api-key>
NUM_INFERENCE_STEPS=<num>
GUIDANCE_SCALE=<num>
WIDTH=<num>
HEIGHT=<num>
NUM_IMAGES_PER_PROMPT=<num>
NEGATIVE_PROMPT=<things-you-dont-want-to-see>
```
For the options see docs at https://huggingface.co/docs/diffusers/v0.9.0/en/api/pipelines/stable_diffusion

## Installation
tbd

## Usage

Sinopsis:
```powershell
python.exe .\src\flux-prompt.py <model>(dev/schnell) <filename> <prompt>
```

Example:
```powershell
python.exe .\src\flux-prompt.py "dev" "golden-retriever" "A fluffy golden retriever"
```

## Examples
![Fluffy Golden Retriever](golden-retriever.png)

## Ideas
- [ ] Keep quantized models in memory for faster response times
