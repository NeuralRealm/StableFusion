import streamlit as st

from stablefusion.scripts.img2img import Img2Img
from stablefusion.scripts.text2img  import Text2Image
from stablefusion.Home import read_model_list
from stablefusion import utils

def app():
    utils.create_base_page()
    if "img2img" in st.session_state and "text2img" not in st.session_state:
        text2img = Text2Image(
            model=st.session_state.img2img.model,
            device=st.session_state.device,
            output_path=st.session_state.output_path,
        )
        st.session_state.text2img = text2img
    with st.form("text2img_model"):
        model = st.selectbox(
                "Which model do you want to use?",
                options=read_model_list(),
            )
        # submit_col, _, clear_col = st.columns(3)
        # with submit_col:
        submit = st.form_submit_button("Load model")
        # with clear_col:
        #    clear = st.form_submit_button("Clear memory")
    # if clear:
    #     clear_memory(preserve="text2img")
    if submit:
        with st.spinner("Loading model..."):
            text2img = Text2Image(
                model=model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
            st.session_state.text2img = text2img
            img2img = Img2Img(
                model=None,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
                text2img_model=text2img.pipeline,
            )
            st.session_state.img2img = img2img
    if "text2img" in st.session_state:
        st.write(f"Current model: {st.session_state.text2img}")
        st.session_state.text2img.app()


if __name__ == "__main__":
    app()
