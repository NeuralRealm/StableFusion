import streamlit as st

from stablefusion import utils
from stablefusion.scripts.pose_editor import OpenPoseEditor
from stablefusion.Home import read_model_list

def app():
    utils.create_base_page()
    with st.form("openpose_editor_form"):
        model = st.selectbox(
            "Which model do you want to use for OpenPose?",
            options=read_model_list()
        )
        submit = st.form_submit_button("Load model")
    if submit:
        st.session_state.openpose_editor = model
        with st.spinner("Loading model..."):
            openpose = OpenPoseEditor(
                model=model,
                device=st.session_state.device,
                output_path=st.session_state.output_path,
            )
            st.session_state.openpose_editor = openpose
    if "openpose_editor" in st.session_state:
        st.write(f"Current model: {st.session_state.openpose_editor}")
        st.session_state.openpose_editor.app()


if __name__ == "__main__":
    app()