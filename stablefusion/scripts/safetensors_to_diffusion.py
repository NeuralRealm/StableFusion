# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conversion script for the LDM checkpoints. """
import ast
import argparse
import os
import streamlit as st
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt
import requests
from stablefusion.utils import base_path

current_path = base_path()

def convert_to_diffusers_model(checkpoint_path, original_config_file, image_size, prediction_type, pipeline_type, extract_ema, scheduler_type, num_in_channels, upcast_attention, from_safetensors, device, stable_unclip, stable_unclip_prior, clip_stats_path, controlnet, checkpoint_name, to_safetensors):

    pipe = load_pipeline_from_original_stable_diffusion_ckpt(
        checkpoint_path=checkpoint_path,
        original_config_file=original_config_file,
        image_size=image_size,
        prediction_type=prediction_type,
        model_type=pipeline_type,
        extract_ema=extract_ema,
        scheduler_type=scheduler_type,
        num_in_channels=num_in_channels,
        upcast_attention=upcast_attention,
        from_safetensors=from_safetensors,
        device=device,
        stable_unclip=stable_unclip,
        stable_unclip_prior=stable_unclip_prior,
        clip_stats_path=clip_stats_path,
        controlnet=controlnet,
    )

    dump_path = "{}/models/diffusion_models/{}".format(current_path, checkpoint_name.split(".")[0])

    if controlnet:
        # only save the controlnet model
        pipe.controlnet.save_pretrained(dump_path, safe_serialization=to_safetensors)
    else:
        pipe.save_pretrained(dump_path, safe_serialization=to_safetensors)
    
    st.success("Your Model Has Been Created!")
    append_created_model(model_name=str(checkpoint_name).split(".")[0])


def download_ckpt_model(checkpoint_link, checkpoint_name):

    try:
        print("Started Downloading the model...")
        response = requests.get(checkpoint_link, stream=True)
        
        with open("{}/models/safetensors_models/{}".format(current_path, checkpoint_name), "wb") as f:
            f.write(response.content)
        
        print("Model Downloading Completed!")
                
    except:
        print("Started Downloading the model...")
        response = requests.get(checkpoint_link, stream=True)

        with open("stablefusion/models/safetensors_models/{}".format(checkpoint_name), "wb") as f:
            f.write(response.content)

        print("Model Downloading Completed!")

def read_model_file():

    try:
        with open('{}/model_list.txt'.format(current_path), 'r') as f:
                contents = f.read()
    except:
        with open('stablefusion/model_list.txt', 'r') as f:
                contents = f.read()

    model_list = ast.literal_eval(contents)

    return model_list


def write_model_list(model_list):
        
        try:
            with open('{}/model_list.txt'.format(current_path), 'w') as f:
                f.write(model_list)
        except:
            with open('stablefusion/model_list.txt', 'w') as f:
                f.write(model_list)


def append_created_model(model_name):
    model_list = read_model_file()
    try:
        apending_list = os.listdir("{}/models/diffusion_models".format(current_path))
    except:
        apending_list = os.listdir("stablefusion/models/diffusion_models")

    for working_item in apending_list:
        if str(working_item).split(".")[-1] == "txt":
            pass
        else:
            if model_name not in model_list:
                try:
                    model_list.append("{}/models/diffusion_models/{}".format(current_path, model_name))
                except:
                    model_list.append("stablefusion/models/diffusion_models/{}".format(model_name))
    
    write_model_list(model_list=str(model_list))
    st.success("Model Added to your List Now you can use this model at your Home Page.")


def convert_safetensor_to_diffusers(original_config_file, image_size, prediction_type, pipeline_type, extract_ema, scheduler_type, num_in_channels, upcast_attention, from_safetensors, device, stable_unclip, stable_unclip_prior, clip_stats_path, controlnet, to_safetensors, checkpoint_name, checkpoint_link, overwrite):

    checkpoint_path = "{}/models/safetensors_models/{}".format(current_path, checkpoint_name)

    try:
        custom_diffusion_model = "{}/models/diffusion_models/{}".format(current_path, str(checkpoint_name).split(".")[0])
    except:
        custom_diffusion_model = "stablefusion/models/diffusion_models/{}".format(str(checkpoint_name).split(".")[0])

    if overwrite is False:
        if not os.path.isfile(checkpoint_path):

            download_ckpt_model(checkpoint_link=checkpoint_link, checkpoint_name=checkpoint_name)

        else:
            st.warning("Using {}".format(checkpoint_path))
    else:

        download_ckpt_model(checkpoint_link=checkpoint_link, checkpoint_name=checkpoint_name)


    if overwrite is False:

        if not os.path.exists(custom_diffusion_model):
            
            convert_to_diffusers_model(checkpoint_name=checkpoint_name, original_config_file=original_config_file, image_size=image_size, prediction_type=prediction_type, extract_ema=extract_ema, scheduler_type=scheduler_type, num_in_channels=num_in_channels, upcast_attention=upcast_attention, from_safetensors=from_safetensors, device=device, stable_unclip=stable_unclip, stable_unclip_prior=stable_unclip_prior, clip_stats_path=clip_stats_path, controlnet=controlnet, to_safetensors=to_safetensors, checkpoint_path=checkpoint_path, pipeline_type=pipeline_type)
        
        else:
            st.warning("Model {} is already present".format(custom_diffusion_model))

    else:
        convert_to_diffusers_model(checkpoint_name=checkpoint_name, original_config_file=original_config_file, image_size=image_size, prediction_type=prediction_type, extract_ema=extract_ema, scheduler_type=scheduler_type, num_in_channels=num_in_channels, upcast_attention=upcast_attention, from_safetensors=from_safetensors, device=device, stable_unclip=stable_unclip, stable_unclip_prior=stable_unclip_prior, clip_stats_path=clip_stats_path, controlnet=controlnet, to_safetensors=to_safetensors, checkpoint_path=checkpoint_path, pipeline_type=pipeline_type)