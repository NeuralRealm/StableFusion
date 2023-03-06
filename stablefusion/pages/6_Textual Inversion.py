import streamlit as st

from stablefusion.scripts.textual_inversion import TextualInversion
from stablefusion import utils
from stablefusion.Home import read_model_list

def app():
    utils.create_base_page()
    with st.form("textual_inversion_form"):
        model = st.selectbox(
            "Which base model do you want to use?",
            options=read_model_list(),
        )
        token_identifier = st.text_input(
            "Token identifier",
            value="<something>"
            if st.session_state.get("textual_inversion_token_identifier") is None
            else st.session_state.textual_inversion_token_identifier,
        )
        embeddings = st.text_input(
            "Embeddings",
            value="https://huggingface.co/sd-concepts-library/axe-tattoo/resolve/main/learned_embeds.bin"
            if st.session_state.get("textual_inversion_embeddings") is None
            else st.session_state.textual_inversion_embeddings,
        )
        # st.file_uploader("Embeddings", type=["pt", "bin"])
        submit = st.form_submit_button("Load model")
    if submit:
        st.session_state.textual_inversion_model = model
        st.session_state.textual_inversion_token_identifier = token_identifier
        st.session_state.textual_inversion_embeddings = embeddings
        with st.spinner("Loading model..."):
            textual_inversion = TextualInversion(
                model=model,
                token_identifier=token_identifier,
                embeddings_url=embeddings,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
            st.session_state.textual_inversion = textual_inversion
    if "textual_inversion" in st.session_state:
        st.write(f"Current model: {st.session_state.textual_inversion}")
        st.session_state.textual_inversion.app()


if __name__ == "__main__":
    app()
