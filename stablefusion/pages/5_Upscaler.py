import streamlit as st

from stablefusion import utils
from stablefusion.models.realesrgan import inference_realesrgan
from stablefusion.scripts.upscaler import Upscaler
from stablefusion.scripts.gfp_gan import GFPGAN


def app():
    utils.create_base_page()
    task = st.selectbox(
        label="Choose Your Upsacler",
        options=[
                    "RealESRGAN",
                    "SD Upscaler",
                    "GFPGAN",
                 ],
    )

    if task == "RealESRGAN":

        model_name = st.selectbox(
        label="Choose Your Model",
        options=[
                    "RealESRGAN_x4plus",
                    "RealESRNet_x4plus",
                    "RealESRGAN_x4plus_anime_6B",
                    "RealESRGAN_x2plus",
                    "realesr-animevideov3",
                    "realesr-general-x4v3",
                    "SD Upscaler",
                    "GFPGAN",
                 ],
    )

        input_image = st.file_uploader(label="Upload The Picture", type=["png", "jpg", "jpeg"])

        col1, col2 = st.columns(2)
        with col1:
            denoise_strength = st.slider(label="Select your Denoise Strength", min_value=0.0 , max_value=1.0, value=0.5, step=0.1, help="Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability.")
            outscale = st.slider(label="Select your final Upsampling scale", min_value=0, max_value=4, value=4, step=1, help="The final upsampling scale of the image")

        with col2:
            face_enhance = st.selectbox(label="Do you want to Inhance Face", options=[True, False], help="Use GFPGAN to enhance face")
            alpha_upsampler = st.selectbox(label="The upsampler for the alpha channels", options=["realesrgan", "bicubic"])

        if st.button("Start Upscaling"):
            with st.spinner("Upscaling The Image..."):
                inference_realesrgan.main(model_name=model_name, outscale=outscale, denoise_strength=denoise_strength, face_enhance=face_enhance, tile=0, tile_pad=10, pre_pad=0, fp32="fp32", gpu_id=None, input_image=input_image, model_path=None)
    

    elif task == "SD Upscaler":
        with st.form("upscaler_model"):
            upscaler_model = st.text_input("Model", "stabilityai/stable-diffusion-x4-upscaler")
            submit = st.form_submit_button("Load model")
        if submit:
            with st.spinner("Loading model..."):
                ups = Upscaler(
                    model=upscaler_model,
                    device=st.session_state.device,
                    output_path=st.session_state.output_path,
                )
                st.session_state.ups = ups
        if "ups" in st.session_state:
            st.write(f"Current model: {st.session_state.ups}")
            st.session_state.ups.app()

    elif task == "GFPGAN":
        with st.spinner("Loading model..."):
            gfpgan = GFPGAN(
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
        gfpgan.app()

if __name__ == "__main__":
    app()
