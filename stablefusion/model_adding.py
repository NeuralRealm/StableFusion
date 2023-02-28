from dataclasses import dataclass
import streamlit as st
import ast
import os


@dataclass
class ModelAdding:

    def read_model_list(self):

        try:
            with open('{}/model_list.txt'.format(os.path.dirname(__file__)), 'r') as f:
                contents = f.read()
        except:
            with open('stablefusion/model_list.txt', 'r') as f:
                contents = f.read()
        model_list = ast.literal_eval(contents)

        return model_list

    def write_model_list(self, model_list):
        
        try:
            with open('{}/model_list.txt'.format(os.path.dirname(__file__)), 'w') as f:
                f.write(model_list)
        except:
            with open('stablefusion/model_list.txt', 'w') as f:
                f.write(model_list)

    def check_models(self, model_name):

        model_list = self.read_model_list()

        if model_name in model_list:
            st.warning("{} already present in the list".format(model_name))

        else:
           model_list.append(model_name)
           self.write_model_list(model_list=str(model_list))
           st.success("Succefully added {} into your list".format(model_name))


    def app(self):
        # upload image
        model_name = st.text_input(label="Enter The Model Name", value="runwayml/stable-diffusion-v1-5")
        
        if st.button("Apply Changes"):
            self.check_models(model_name=model_name)
        
