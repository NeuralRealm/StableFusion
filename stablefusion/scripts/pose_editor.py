import gc
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Union
import random
import requests
import streamlit as st
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from loguru import logger
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from controlnet_aux import OpenposeDetector, HEDdetector, MLSDdetector
from stablefusion import utils
import cv2
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from transformers import pipeline
from stablefusion.scripts.pose_html import html_component


@dataclass
class OpenPoseEditor:
    model: Optional[str] = None
    device: Optional[str] = None
    output_path: Optional[str] = None

    def __str__(self) -> str:
        return f"BaseModel(model={self.model}, device={self.device}, output_path={self.output_path})"
    
    
    def __post_init__(self):
        self.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_auth_token=utils.use_auth_token(),
            )
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model,
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_auth_token=utils.use_auth_token(),
        )

        self.pipeline.to(self.device)
        self.pipeline.safety_checker = utils.no_safety_checker
        self._compatible_schedulers = self.pipeline.scheduler.compatibles
        self.scheduler_config = self.pipeline.scheduler.config
        self.compatible_schedulers = {scheduler.__name__: scheduler for scheduler in self._compatible_schedulers}

        if self.device == "mps":
            self.pipeline.enable_attention_slicing()
            # warmup
            url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
            response = requests.get(url)
            init_image = Image.open(BytesIO(response.content)).convert("RGB")
            init_image.thumbnail((768, 768))
            prompt = "A fantasy landscape, trending on artstation"
            _ = self.pipeline(
                prompt=prompt,
                image=init_image,
                strength=0.75,
                guidance_scale=7.5,
                num_inference_steps=2,
            )

    def _set_scheduler(self, scheduler_name):
        scheduler = self.compatible_schedulers[scheduler_name].from_config(self.scheduler_config)
        self.pipeline.scheduler = scheduler

    def generate_image(
        self, prompt, image, negative_prompt, scheduler, num_images, guidance_scale, steps, seed, height, width
    ):
        self._set_scheduler(scheduler)
        logger.info(self.pipeline.scheduler)
        if self.device == "mps":
            generator = torch.manual_seed(seed)
            num_images = 1
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        num_images = int(num_images)
        output_images = self.pipeline(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
            height=height,
            width=width
        ).images
        torch.cuda.empty_cache()
        gc.collect()
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "scheduler": scheduler,
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "seed": seed,
        }
        metadata = json.dumps(metadata)
        _metadata = PngInfo()
        _metadata.add_text("img2img", metadata)

        utils.save_images(
            images=output_images,
            module="controlnet",
            metadata=metadata,
            output_path=self.output_path,
        )
        return output_images, _metadata

    def app(self):
        available_schedulers = list(self.compatible_schedulers.keys())
        # if EulerAncestralDiscreteScheduler is available in available_schedulers, move it to the first position
        if "EulerAncestralDiscreteScheduler" in available_schedulers:
            available_schedulers.insert(
                0, available_schedulers.pop(available_schedulers.index("EulerAncestralDiscreteScheduler"))
            )
        
        html_component()

        input_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if input_image is not None:

            input_image = Image.open(input_image).convert("RGB")

            st.image(input_image, use_column_width=True)


        # with st.form(key="img2img"):
        col1, col2 = st.columns(2)
        with col1:
            prompt = st.text_area("Prompt", "", help="Prompt to guide image generation")
        with col2:
            negative_prompt = st.text_area("Negative Prompt", "", help="The prompt not to guide image generation. Write things that you dont want to see in the image.")

        scheduler = st.sidebar.selectbox("Scheduler", available_schedulers, index=0, help="Scheduler(Sampler) to use for generation")
        image_height = st.sidebar.slider("Image height", 128, 1024, 512, 128, help="The height in pixels of the generated image.")
        image_width = st.sidebar.slider("Image width", 128, 1024, 512, 128, help="The width in pixels of the generated image.")
        guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 40.0, 7.5, 0.5, help="Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.")
        num_images = st.sidebar.slider("Number of images per prompt", 1, 30, 1, 1, help="Number of images you want to generate. More images requires more time and uses more GPU memory.")
        steps = st.sidebar.slider("Steps", 1, 150, 50, 1, help="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.")
        seed_choice = st.sidebar.selectbox("Do you want a random seed", options=["Yes", "No"])
        if seed_choice == "Yes":
            seed = random.randint(0, 9999999)
        else:
            seed = st.sidebar.number_input(
                "Seed",
                value=42,
                step=1,
                help="Random seed. Change for different results using same parameters.",
            )
        sub_col, download_col = st.columns(2)
        with sub_col:
            submit = st.button("Generate")

        if submit:
            with st.spinner("Generating images..."):
                output_images, metadata = self.generate_image(
                    prompt=prompt,
                    image=input_image,
                    negative_prompt=negative_prompt,
                    scheduler=scheduler,
                    num_images=num_images,
                    guidance_scale=guidance_scale,
                    steps=steps,
                    seed=seed,
                    height=image_height,
                    width=image_width
                )

            utils.display_and_download_images(output_images, metadata, download_col)
