import os
import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from typing import Iterable

from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from aura_sr import AuraSR

# --- # Device and CUDA Setup Check ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("Using device:", device)

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red, 
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()


# --- Main Model Initialization ---
MAX_SEED = np.iinfo(np.int32).max
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")

# --- Load Adapters ---
pipe.load_lora_weights("prithivMLmods/Kontext-Top-Down-View", weight_name="Kontext-Top-Down-View.safetensors", adapter_name="top-down")
pipe.load_lora_weights("prithivMLmods/Kontext-Bottom-Up-View", weight_name="Kontext-Bottom-Up-View.safetensors", adapter_name="bottom-up")
pipe.load_lora_weights("prithivMLmods/Kontext-CAM-Left-View", weight_name="Kontext-CAM-Left-View.safetensors", adapter_name="left-view")
pipe.load_lora_weights("prithivMLmods/Kontext-CAM-Right-View", weight_name="Kontext-CAM-Right-View.safetensors", adapter_name="right-view")
pipe.load_lora_weights("starsfriday/Kontext-Remover-General-LoRA", weight_name="kontext_remove.safetensors", adapter_name="kontext-remove")

# --- Upscaler Initialization ---
aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

@spaces.GPU
def infer(input_image, prompt, lora_adapter, upscale_image, seed=42, randomize_seed=False, guidance_scale=2.5, steps=28, progress=gr.Progress(track_tqdm=True)):
    """
    Perform image editing and optional upscaling, returning the final image.
    """
    if not input_image:
        raise gr.Error("Please upload an image for editing.")

    if lora_adapter == "Kontext-Top-Down-View":
        pipe.set_adapters(["top-down"], adapter_weights=[1.0])
    elif lora_adapter == "Kontext-Bottom-Up-View":
        pipe.set_adapters(["bottom-up"], adapter_weights=[1.0])
    elif lora_adapter == "Kontext-CAM-Left-View":
        pipe.set_adapters(["left-view"], adapter_weights=[1.0])
    elif lora_adapter == "Kontext-CAM-Right-View":
        pipe.set_adapters(["right-view"], adapter_weights=[1.0])
    elif lora_adapter == "Kontext-Remover":
        pipe.set_adapters(["kontext-remove"], adapter_weights=[1.0])
        
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    original_image = input_image.copy().convert("RGB")
    
    image = pipe(
        image=original_image, 
        prompt=prompt,
        guidance_scale=guidance_scale,
        width = original_image.size[0],
        height = original_image.size[1],
        num_inference_steps=steps,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]

    if upscale_image:
        progress(0.8, desc="Upscaling image...")
        image = aura_sr.upscale_4x(image)

    return image, seed, gr.Button(visible=True)

@spaces.GPU
def infer_example(input_image, prompt, lora_adapter):
    """
    Wrapper function for gr.Examples.
    """
    image, seed, _ = infer(input_image, prompt, lora_adapter, upscale_image=False)
    return image, seed

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
#main-title h1 {font-size: 2.2em !important;}
"""

with gr.Blocks() as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **Kontext-Photo-Mate-v2**", elem_id="main-title")
        gr.Markdown("Image manipulation with FLUX.1 Kontext adapters. [How to Use](https://huggingface.co/spaces/prithivMLmods/Photo-Mate-i2i/discussions/2)")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Image", type="pil", height=290)
                
                prompt = gr.Text(
                    label="Edit Prompt",
                    show_label=True,
                    placeholder="e.g., transform into anime..",
                )

                run_button = gr.Button("Edit Image", variant="primary")

                with gr.Accordion("Advanced Settings", open=False):
                    
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=10,
                        step=0.1,
                        value=2.5,
                    )       
                    
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=30,
                        value=28,
                        step=1
                    )
                    
            with gr.Column():
                output_image = gr.Image(label="Output Image", interactive=False, format="png", height=355)
                reuse_button = gr.Button("Reuse this image", visible=False)

                with gr.Row():
                    lora_adapter = gr.Dropdown(
                        label="Chosen LoRA",
                        choices=["Kontext-Top-Down-View", "Kontext-Remover", "Kontext-Bottom-Up-View", "Kontext-CAM-Left-View", "Kontext-CAM-Right-View"],
                        value="Kontext-Top-Down-View"
                    )
                    
                with gr.Row():
                    upscale_checkbox = gr.Checkbox(label="Upscale the final image", value=False)

        gr.Examples(
            examples=[
                ["examples/1.jpeg", "[photo content], recreate the scene from a top-down perspective. Maintain all visual proportions, lighting consistency, and realistic spatial relationships. Ensure the background, textures, and environmental shadows remain naturally aligned from this elevated angle.", "Kontext-Top-Down-View"],
                ["examples/2.jpg", "Remove the apples under the kitten's body", "Kontext-Remover"],
                ["examples/3.jpg", "[photo content], recreate the scene from a bottom-up perspective. Preserve accurate depth, scale, and lighting direction to enhance realism. Ensure the background sky or floor elements adjust naturally to the new angle, maintaining authentic shadowing and perspective.", "Kontext-Bottom-Up-View"],
                ["examples/4.jpg", "[photo content], render the image from the left-side perspective, keeping consistent lighting, textures, and proportions. Maintain the realism of all surrounding elements while revealing previously unseen left-side details consistent with the object’s or scene’s structure", "Kontext-CAM-Left-View"],
                ["examples/5.jpg", "[photo content], generate the right-side perspective of the scene. Ensure natural lighting, accurate geometry, and realistic textures. Maintain harmony with the original image’s environment, shadows, and visual tone while providing the right-side visual continuation.", "Kontext-CAM-Right-View"],
            ],
            inputs=[input_image, prompt, lora_adapter],
            outputs=[output_image, seed],
            fn=infer_example,
            cache_examples=False,
            label="Examples"
        )
            
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[input_image, prompt, lora_adapter, upscale_checkbox, seed, randomize_seed, guidance_scale, steps],
        outputs=[output_image, seed, reuse_button]
    )
    
    reuse_button.click(
        fn=lambda x: x,
        inputs=[output_image],
        outputs=[input_image]
    )

demo.launch(theme=orange_red_theme, css=css, mcp_server=True, ssr_mode=False, show_error=True)
