import streamlit as st

from stablefusion import utils
from stablefusion.scripts.gfp_gan import GFPGAN
from stablefusion.scripts.image_info import ImageInfo
from stablefusion.scripts.interrogator import ImageInterrogator
from stablefusion.scripts.upscaler import Upscaler
from stablefusion.scripts.model_adding import ModelAdding
from stablefusion.scripts.model_removing import ModelRemoving


def app():
    utils.create_base_page()
    task = st.selectbox(
        "Choose a utility",
        [
            "ImageInfo",
            "Model Adding",
            "Model Removing",
            "SD Upscaler",
            "GFPGAN",
            "CLIP Interrogator",
        ],
    )
    if task == "ImageInfo":
        ImageInfo().app()
    elif task == "Model Adding":
        ModelAdding().app()
    elif task == "Model Removing":
        ModelRemoving().app()
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
    elif task == "CLIP Interrogator":
        interrogator = ImageInterrogator(
            device=st.session_state.device,
            output_path=st.session_state.output_path,
        )
        interrogator.app()


if __name__ == "__main__":
    app()
