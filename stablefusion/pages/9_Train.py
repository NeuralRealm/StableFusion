import streamlit as st
from stablefusion import utils
from stablefusion.scripts.dreambooth import train_dreambooth
from stablefusion.Home import read_model_list
import json
import os


def dump_concept_file(instance_prompt, class_prompt, instance_data_dir, class_data_dir):

    concepts_list = [
    {
        "instance_prompt":      instance_prompt,
        "class_prompt":         class_prompt,
        "instance_data_dir":    "{}/trainner_assets/instance_data/{}".format(utils.base_path(), instance_data_dir),
        "class_data_dir":       "{}/trainner_assets/class_data/{}".format(utils.base_path(), class_data_dir)
    },
    ]
    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    with open("{}/trainner_assets/concepts_list.json".format(utils.base_path()), "w") as f:
        json.dump(concepts_list, f, indent=4)


def app():
    utils.create_base_page()
    task = st.selectbox(
        "Choose a Convertor",
        [
            "Dreambooth",
        ],
    )
    if task == "Dreambooth":

        with st.form("Train With Dreambooth"):

            col1, col2 = st.columns(2)
            with col1:
                base_model = st.selectbox(label="Name of the Base Model", options=read_model_list(), help="Path to pretrained model or model identifier from huggingface.co/models.")
                instance_prompt = st.text_input(label="The prompt with identifier specifying the instance", value="photo of elon musk person", help="replace elon musk name with your object or person name")
                class_prompt = st.text_input(label="The prompt to specify images in the same class as provided instance images.", value="photo of a person", help="replace person with object if you are taining for any kind of object")
                instance_data_dir_name = st.text_input(label="The prompt with identifier specifying the instance(folder): ", value="elon musk", help="A folder containing the training data of instance images")
                class_data_dir_name = st.text_input(label="A Name of the training data of class images(folder): ", value="person", help="A folder containing the training data of class images.")
                output_dir_name = st.text_input(label="Name of your Trainned model(folder): ", value="elon musk", help="The output directory name where the model predictions and checkpoints will be written.")
                seed = st.number_input(label="Enter The Seed", value=1337, step=1, help="A seed for reproducible training.")

            with col2:
                resolution = st.number_input(label="Enter The Image Resolution: ", value=512, step=1, help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
                train_batch_size = st.number_input(label="Train Batch Size", value=4, step=1, help="Batch size (per device) for sampling images.")
                learning_rate = st.number_input(label="Learning Rate", value=6, help="Initial learning rate (after the potential warmup period) to use. float(1) / float(10**8) for 1e-8")
                num_class_images = st.number_input(label="Number of Class Images", value=100, step=1, help="Minimal class images for prior preservation loss. If there are not enough images already present in class_data_dir, additional images will be sampled with class_prompt.")
                sample_batch_size = st.number_input(label="Sample Batch Size: ", value=4, help="Batch size (per device) for sampling images.")
                max_train_steps = st.number_input(label="Max Train Steps: ", value=20, help="Total number of training steps to perform.")
                save_interval = st.number_input(label="Save Interal Steps: ", value=10000, help="Save weights every N steps.")
            
            st.subheader("Advance Settings")
            st.text("Don't Change anything if you are not sure about it.")
            col3, col4 = st.columns(2)
            with col3:
                revision = st.text_input(label="Revision of pretrained model identifier: ", value=None)
                prior_loss_weight_condition = st.selectbox(label="Want to use prior_loss_weight:  ", options=[True, False])
                if prior_loss_weight_condition:
                    prior_loss_weight = st.slider(label="Prior Loss Weight: ",value=1.0, max_value=10.0, help="The weight of prior preservation loss.")
                else:
                    prior_loss_weight = 0

                train_text_encoder = st.selectbox("Whether to train the text encoder", options=[True, False])
                log_interval = st.number_input(label="Save Log Interval: ", value=10, help="Log every N steps.")
                tokenizer_name = st.text_input("Enter the tokenizer name", value=None, help="Pretrained tokenizer name or path if not the same as model_name")

            with col4:
                use_8bit_adam = st.selectbox(label="Want to use 8bit adam: ", options=[True, False], help="Whether or not to use 8-bit Adam from bitsandbytes.")
                gradient_accumulation_steps = st.slider(label="Gradient Accumulation steps", value=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
                lr_scheduler = st.selectbox(label="Select Your Learning Schedular: ", options=["constant", "cosine", "cosine_with_restarts", "polynomial","linear", "constant_with_warmup"])
                lr_warmup_steps = st.number_input(label="Learning Rate, Warmup Steps: ", value=50, step=1, help="Number of steps for the warmup in the lr scheduler.")
                mixed_precision = st.selectbox(label="Whether to use mixed precision", options=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >=1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.")
            
            
            submit = st.form_submit_button("Start Trainning")
            
            if submit:
                with st.spinner("Trainning..."):

                    output_dir = "{}/trainner_assets/stable_diffusion_weights/{}".format(utils.base_path(), output_dir_name)

                    learning_rate = 1 * 10**(-learning_rate)

                    dump_concept_file(instance_prompt=instance_prompt, class_prompt=class_prompt, instance_data_dir=instance_data_dir_name, class_data_dir=class_data_dir_name)
                    train_dreambooth.main(pretrained_model_name_or_path=base_model, revision=revision, output_dir=output_dir, with_prior_preservation=prior_loss_weight, prior_loss_weight=prior_loss_weight, seed=seed, resolution=resolution, train_batch_size=train_batch_size, train_text_encoder=train_text_encoder, mixed_precision=mixed_precision, use_8bit_adam=use_8bit_adam, gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=learning_rate, lr_scheduler=lr_scheduler, lr_warmup_steps=lr_warmup_steps, num_class_images=num_class_images, sample_batch_size=sample_batch_size, max_train_steps=max_train_steps, save_interval=save_interval, save_sample_prompt=None, log_interval=log_interval, tokenizer_name=tokenizer_name)
                    


if __name__ == "__main__":
    app()
