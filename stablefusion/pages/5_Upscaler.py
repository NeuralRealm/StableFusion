import streamlit as st

from stablefusion import utils
from stablefusion.models.realesrgan import inference_realesrgan


def app():
    utils.create_base_page()
    model_name = st.selectbox(
        label="Choose Your Model",
        options=[
                    "RealESRGAN_x4plus",
                    "RealESRNet_x4plus",
                    "RealESRGAN_x4plus_anime_6B",
                    "RealESRGAN_x2plus",
                    "realesr-animevideov3",
                    "realesr-general-x4v3",
                 ],
    )
    input_image = st.file_uploader(label="Upload The Picture", type=["png", "jpg", "jpeg"])

    col1, col2 = st.columns(2)
    with col1:
        denoise_strength = st.slider(label="Select your Denoise Strength", min_value=0.0 , max_value=1.0, value=0.5, step=0.1, help="Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability.")
        outscale = st.slider(label="Select your final Upsampling scale", min_value=0, max_value=10, value=4, step=1, help="The final upsampling scale of the image")

    with col2:
        face_enhance = st.selectbox(label="Do you want to Inhance Face", options=[True, False], help="Use GFPGAN to enhance face")
        alpha_upsampler = st.selectbox(label="The upsampler for the alpha channels", options=["realesrgan", "bicubic"])

    if st.button("Start Upscaling"):
        with st.spinner("Upscaling The Image..."):
            inference_realesrgan.main(model_name=model_name, outscale=outscale, denoise_strength=denoise_strength, face_enhance=face_enhance, tile=0, tile_pad=10, pre_pad=0, fp32="fp32", gpu_id=None, input_image=input_image, model_path=None)
    


if __name__ == "__main__":
    app()
