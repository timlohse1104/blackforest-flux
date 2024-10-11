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
DEBUG_SESSION=<bool>
MANUAL_SEED=<num>
```
For the options see docs at https://huggingface.co/docs/diffusers/v0.9.0/en/api/pipelines/stable_diffusion

## Installation
tbd

## Usage

Sinopsis:
```powershell
python.exe .\src\flux-session.py <model>(dev/schnell)
```

Example:
```powershell
python.exe .\src\flux-prompt.py "dev"
```

## Examples
![Fluffy Golden Retriever](golden-retriever.png)

## Ideas
- [x] Use quantization as performance boost
- [x] Create a long living session so user is able to create multiple files
- [x] Make pipe options customizable from .env file
- [x] Continue session with last prompt
- [x] Make pipe options customizable from within session
- [x] Manual seed support for reproducibility
- [ ] Create UI to input prompt and filename (maybe set pipe options)
- [ ] Extend UI to view all created images
