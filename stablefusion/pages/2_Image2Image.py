import streamlit as st

from stablefusion.scripts.img2img import Img2Img
from stablefusion.scripts.text2img import Text2Image
from stablefusion.Home import read_model_list
from stablefusion import utils

def app():
    utils.create_base_page()
    if "text2img" in st.session_state and "img2img" not in st.session_state:
        img2img = Img2Img(
            model=None,
            device=st.session_state.device,
            output_path=st.session_state.output_path,
            text2img_model=st.session_state.text2img.pipeline,
        )
        st.session_state.img2img = img2img
    with st.form("img2img_model"):
        model = st.selectbox(
                "Which model do you want to use?",
                options=read_model_list(),
            )
        submit = st.form_submit_button("Load model")
    if submit:
        with st.spinner("Loading model..."):
            img2img = Img2Img(
                model=model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
            st.session_state.img2img = img2img
            text2img = Text2Image(
                model=model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
            st.session_state.text2img = text2img
    if "img2img" in st.session_state:
        st.write(f"Current model: {st.session_state.img2img}")
        st.session_state.img2img.app()


if __name__ == "__main__":
    app()
