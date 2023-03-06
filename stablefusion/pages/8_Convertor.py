import streamlit as st

from stablefusion import utils
from stablefusion.scripts.gfp_gan import GFPGAN
from stablefusion.scripts.image_info import ImageInfo
from stablefusion.scripts.interrogator import ImageInterrogator
from stablefusion.scripts.upscaler import Upscaler
from stablefusion.scripts.model_adding import ModelAdding
from stablefusion.scripts.model_removing import ModelRemoving
from stablefusion.scripts.ckpt_to_diffusion import convert_ckpt_to_diffusion
from stablefusion.scripts.safetensors_to_diffusion import convert_safetensor_to_diffusers


def app():
    utils.create_base_page()
    task = st.selectbox(
        "Choose a Convertor",
        [
            "CKPT to Diffusers",
            "Safetensors to Diffusers",
        ],
    )
    if task == "CKPT to Diffusers":

        with st.form("Convert Your CKPT to Diffusers"):
            ckpt_model_name = st.text_input(label="Name of the CKPT Model", value="NameOfYourModel.ckpt")
            ckpt_model = st.text_input(label="Download Link of CKPT Model path: ", value="https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.0-pruned-fp16.ckpt", help="Path to the checkpoint to convert.")
            overwrite_mode = st.selectbox("Do You want to overwrite file: ", options=[False, True], help="If the ckpt files with that same name is present do you want to overwrite the file or want to use that file instead of downloading it again?")            
            st.subheader("Advance Settings")
            st.text("Don't Change anything if you are not sure about it.")

            config_file = st.text_input(label="Enter The Config File: ", value=None, help="The YAML config file corresponding to the original architecture.")
            
            col1, col2 = st.columns(2)

            with col1:
                num_in_channels = st.text_input(label="Enter The Number Of Channels", value=None, help="The number of input channels. If `None` number of input channels will be automatically inferred.")
                scheduler_type = st.selectbox(label="Select the Scheduler: ", options=['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm'], help="Type of scheduler to use.")
                pipeline_type = st.text_input(label="Enter the pipeline type: ", value=None, help="The pipeline type. If `None` pipeline will be automatically inferred.")
            
            with col2:
                image_size = st.selectbox(label="Image Size", options=[None, "512", "768"], help="The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Siffusion v2 Base. Use 768 for Stable Diffusion v2.")
                prediction_type = st.selectbox(label="Select your prediction type", options=[None, "epsilon", "v-prediction"])
                extract_ema = st.selectbox(label="Extract the EMA", options=[True, False], help="Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning.")
            
            device = None
            
            submit = st.form_submit_button("Start Converting")
            
            if submit:
                with st.spinner("Converting..."):
                    convert_ckpt_to_diffusion(device=device, checkpoint_link=ckpt_model, checkpoint_name=ckpt_model_name, num_in_channels=num_in_channels, scheduler_type=scheduler_type, pipeline_type=pipeline_type, extract_ema=extract_ema, dump_path=" ", image_size=image_size, original_config_file=config_file, prediction_type=prediction_type, overwrite_file=overwrite_mode)
        

    elif task == "Safetensors to Diffusers":

        with st.form("Convert Your Safetensors to Diffusers"):
            stabletensor_model_name = st.text_input(label="Name of the Safetensors Model", value="NameOfYourModel.safetensors")
            stabletensor_model = st.text_input(label="Download Link of Safetensors Model path: ", value="https://civitai.com/api/download/models/4007?type=Model&format=SafeTensor", help="Path to the checkpoint to convert.")
            overwrite_mode = st.selectbox("Do You want to overwrite file: ", options=[False, True], help="If the Stabletensor files with that same name is present do you want to overwrite the file or want to use that file instead of downloading it again?")            
            st.subheader("Advance Settings")
            st.text("Don't Change anything if you are not sure about it.")

            config_file = st.text_input(label="Enter The Config File: ", value=None, help="The YAML config file corresponding to the original architecture.")
            
            col1, col2 = st.columns(2)

            with col1:
                num_in_channels = st.text_input(label="Enter The Number Of Channels", value=None, help="The number of input channels. If `None` number of input channels will be automatically inferred.")
                scheduler_type = st.selectbox(label="Select the Scheduler: ", options=['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm'], help="Type of scheduler to use.")
                pipeline_type = st.text_input(label="Enter the pipeline type: ", value=None, help="The pipeline type. If `None` pipeline will be automatically inferred.")
            
            with col2:
                image_size = st.selectbox(label="Image Size", options=[None, "512", "768"], help="The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Siffusion v2 Base. Use 768 for Stable Diffusion v2.")
                prediction_type = st.selectbox(label="Select your prediction type", options=[None, "epsilon", "v-prediction"])
                extract_ema = st.selectbox(label="Extract the EMA", options=[True, False], help="Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning.")
            
            device = None
            
            submit = st.form_submit_button("Start Converting")
            
            if submit:
                with st.spinner("Converting..."):
                    if config_file == "None":
                        config_file = None
                    
                    if num_in_channels == "None":
                        num_in_channels = None
                    
                    if pipeline_type == "None":
                        pipeline_type = None
                    convert_safetensor_to_diffusers(original_config_file=config_file, image_size=image_size, prediction_type=prediction_type, pipeline_type=pipeline_type, extract_ema=extract_ema, scheduler_type=scheduler_type, num_in_channels=num_in_channels, upcast_attention=False, from_safetensors=True, device=device, stable_unclip=None, stable_unclip_prior=None, clip_stats_path=None, controlnet=None, to_safetensors=None, checkpoint_name=stabletensor_model_name, checkpoint_link=stabletensor_model, overwrite=overwrite_mode)
                    


if __name__ == "__main__":
    app()