import streamlit as st
import numpy as np
import os
import string
from generating_baby_names import main


st.beta_set_page_config(
    page_title="Name generator",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# def model_selector(folder_path='.'):
#     filenames = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)


def process_names(names):
    return [name for name in names.split("\n")]


def validate_letter(letter):
    if letter != 'Any':
        if letter.isalpha():
            return letter
        else:
            print("Should be a letter of the English alphabet.")
    else:
        return letter

letters = list(string.ascii_lowercase)
letters.insert(0, 'Any')

input_container, output_container = st.beta_columns(2)
with input_container:
    st.subheader("Enter names one after the other. You can also copy and paste. Please make sure you don't enclose them in single or double quotes.")
    names = st.text_area("Paste or type in names here.")
    names = process_names(names)
    
    st.subheader("First letter")
    first_letter = st.selectbox('Select First letter', letters)
    validate_letter(first_letter)
    if first_letter == 'Any':
        first_letter = None

    st.subheader("Second letter")
    second_letter = st.selectbox('Select Second letter', letters)
    validate_letter(second_letter)
    if second_letter == 'Any':
        second_letter = None

    st.subheader("Minimum name length")
    # minimum_name_length = st.selectbox('Select a minimum length.', [i for i in range(2, 21)])
    minimum_name_length = st.slider('Select a minimum length.', min_value=2, max_value=20)

    st.subheader("Names in output?")
    # num_names = st.selectbox("Enter the number of names you want to see in the output. Default value is 10.", [i for i in range(1, 101)])
    num_names = st.slider("Enter the number of names you want to see in the output. Default value is 10.", min_value=10, max_value=200)

    st.subheader("Epochs")
    epochs = st.selectbox("Epochs are the number of times your model trains on the input data. More number of epochs might give you better results, but that is not always the case. It is best to play around with this parameter. To ensure that no one tries to give a million epochs even by accident, I have restricted my epoch count to 4000. This will still take some time to run.", [100, 200, 400, 800, 1000, 2000, 4000])    
    

    button = st.button("Train model and show results.")
    if button:
        st.write("Scroll up and you should see the results to the right.")
        with output_container:
            with st.spinner("Training model..."):
                    generated_names = main(names, first_letter, second_letter, minimum_name_length, num_names, epochs)
            for name in generated_names:
                st.text(name.replace(".", ""))
        st.balloons()
    

