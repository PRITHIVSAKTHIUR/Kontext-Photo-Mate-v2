# **Kontext-Photo-Mate-v2**

> **Kontext-Photo-Mate-v2** is an advanced image manipulation application built on top of the FLUX.1 Kontext AI model, featuring multiple LoRA adapters for fine-grained perspective and object editing. This tool allows users to upload an image, provide textual prompts for creative or corrective edits, and optionally upscale the result for enhanced detail, all via a Gradio web interface with a custom orange-red theme.

<img width="1611" height="1166" alt="C6ONX-_Yyr4Ot0QEkDmCP" src="https://github.com/user-attachments/assets/cf2c2241-a8ea-4fb5-8f0e-9badc4aaa339" />
<img width="1447" height="1066" alt="VjaP0Oy7dV35SgAV4026Q" src="https://github.com/user-attachments/assets/e8b4d03c-f51a-4978-a7a8-669cb5bb522a" />

## Features

- Image editing driven by powerful FLUX.1 Kontext adapters targeting different views: top-down, bottom-up, left/right camera angles, and object removal.
- Fine control over editing with advanced settings including random or fixed seeds, guidance scales, and inference steps.
- Optional 4x upscaling using AuraSR.
- Interactive before/after image slider for comparing edits.
- Reuse generated images for iterative editing.
- Smooth UI experience with Gradio and a visually appealing orange-red theme.

## Getting Started

### Installation

To run Kontext-Photo-Mate-v2 locally, install required dependencies using:

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
- git+https://github.com/huggingface/accelerate.git
- git+https://github.com/huggingface/diffusers.git
- git+https://github.com/huggingface/peft.git
- gradio-imageslider
- huggingface_hub
- sentencepiece
- transformers
- torchvision
- aura-sr
- spaces
- torch

### Repository

Clone the repository:

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Kontext-Photo-Mate-v2.git
cd Kontext-Photo-Mate-v2
```

## Usage

Run the Gradio interface script (e.g., `python app.py`) to launch the web UI.

### Interface Overview

- **Upload Image:** Upload an image to be edited.
- **Edit Prompt:** Enter natural language instructions for modification, e.g., "Remove glasses", or detailed perspective changes.
- **LoRA Adapter:** Select the adapter that targets specific contextual transformations:  
  - Kontext-Top-Down-View  
  - Kontext-Bottom-Up-View  
  - Kontext-CAM-Left-View  
  - Kontext-CAM-Right-View  
  - Kontext-Remover  
- **Advanced Settings:** Adjust seed, randomization, guidance scale, and steps for inference stability and creativity.
- **Upscale Option:** Enable to enhance the resulting image by 4x using AuraSR.
- **Before/After Slider:** Compare original and edited images interactively.
- **Reuse Button:** Use the generated image as a new input for iterative edits.

### Examples

Example prompts demonstrate various use cases such as scene recreation from different perspectives and object removal:

- "Recreate the scene from a top-down perspective. Maintain all visual proportions, lighting consistency, and realistic spatial relationships."
- "Remove the apples under the kitten's body."
- "Render the image from the left-side perspective, keeping consistent lighting, textures, and proportions."

## Model Details

- The system uses the FLUX.1 Kontext pipeline with multiple LoRA weights loaded to enable diverse editing perspectives.
- Employs AuraSR for upscaling.
- Device setup supports CUDA GPUs if available.

## Developer Notes

- The core inference performs conditional image editing with text prompt guidance and LoRA adapter selection.
- Advanced Gradio callbacks enable synchronized UI controls between input/output and buttons.
- Custom theme "OrangeRedTheme" defines styling consistent with an orange-red palette enhancing visual appeal.
