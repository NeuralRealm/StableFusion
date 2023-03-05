import streamlit as st

from stablefusion import utils
from stablefusion.scripts.inpainting import Inpainting
from stablefusion.Home import read_model_list

def app():
    utils.create_base_page()
    with st.form("inpainting_model_form"):
        model = st.selectbox(
            "Which model do you want to use for inpainting?",
            options=read_model_list()
        )
        pipeline = st.selectbox(label="Select Your Pipeline: ", options=["StableDiffusionInpaintPipelineLegacy" ,"StableDiffusionInpaintPipeline"])
        submit = st.form_submit_button("Load model")
    if submit:
        st.session_state.inpainting_model = model
        with st.spinner("Loading model..."):
            inpainting = Inpainting(
                model=model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
                pipeline_select=pipeline
            )
            st.session_state.inpainting = inpainting
    if "inpainting" in st.session_state:
        st.write(f"Current model: {st.session_state.inpainting}")
        st.session_state.inpainting.app()


if __name__ == "__main__":
    app()