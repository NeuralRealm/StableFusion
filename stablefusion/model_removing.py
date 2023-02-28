from dataclasses import dataclass
import streamlit as st
import ast
import os


@dataclass
class ModelRemoving:

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

        if model_name not in model_list:
            st.warning("{} not present in the list".format(model_name))

        else:
           model_list.remove(model_name)
           self.write_model_list(model_list=str(model_list))
           st.success("Succefully Removed {} into your list".format(model_name))


    def app(self):
        # upload image
        model_name = st.selectbox(label="Enter The Model Name", options=self.read_model_list())
        
        if st.button("Apply Changes"):
            self.check_models(model_name=model_name)
        
