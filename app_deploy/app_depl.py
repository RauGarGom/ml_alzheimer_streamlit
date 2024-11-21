import sys
import os
import numpy as np
import streamlit as st
import plotly.express as px
# import numpy as np
#  sys.path.append(os.path.abspath(r'C:\Users\raulg\Documents\THEBRIDGE_DS\0.-Repo_Git\ml_alzheimer_class\src'))
# sys.path.append(os.path.relpath('/src'))
from utils import model_prediction, img_model_prediction


###Global variables
if "result" not in st.session_state:
    st.session_state['result'] = ''
if "chart" not in st.session_state:
    st.session_state.chart = None
if 'class_result' not in st.session_state:
    st.session_state.class_result = None #[0,0,0,0]
mapping_class = {
        0: 'No Alzheimer',
        1: 'Alzheimer',
    }
mapping_img = {
        0: 'Non Demented',
        1: 'Very Mild Demented',
        2: 'Mild Demented',
        3: 'Moderate Demented'
    }
with st.sidebar:
    st.title("Charts of the predictions")
    if st.session_state['chart'] == None:
        st.write("No chart to display yet.")
        

# Title of the app and header
st.title('Alzheimer diagnosis tool')
st.subheader('Medical tool for Alzheimer diagnosis by Raúl García')
st.subheader('')



st.header('Preliminary form')
### CLASS - Inputs
funct_assess = st.slider("What's the patient's Functional Assessment Scoring?:",0,30)

col1,col2 = st.columns(2,gap='medium')
with col1:
    memory = st.pills(
        "Does the patient present memory inefficiencies?",
        ("Yes", "No"),key=0,default='No')
    mmse = st.slider("What's the patient's MMSE scoring",0,10)


with col2:
    behav = st.pills(
        "Does the patient present behavioral issues?",
        ("Yes", "No"),key=1,default='No')
    adl = st.slider(
    "How does the patient perceives their daily life?",0,10)


### CLASS : Model and charts
def results():
    st.session_state['result'] = model_prediction(mmse,funct_assess,memory,behav,adl)

if st.button('Run prediction'):
    st.session_state.class_result = model_prediction(mmse, funct_assess, memory, behav, adl)
    class_result = st.session_state.class_result
    if class_result[3] == 1:
        st.error(class_result[1])
    elif class_result[3]==0:
        st.success(class_result[1])
    else:
        st.warning(class_result[1])
    chart = px.pie(
        values=class_result[2].flatten(),
        names=mapping_class.values(),
        title='Probabilities of class prediction'
    )
    st.session_state['chart'] = chart
    with st.sidebar:
        st.plotly_chart(chart)

if st.session_state['class_result'] and st.session_state['class_result'][3] > 0:
    st.header('MRI scan prediction')
    uploaded_img = st.file_uploader("Upload the patient's MRI scan", type=["jpg", "jpeg", "png"])
    # enable = st.checkbox("Enable camera")
    # uploaded_img = st.camera_input("Picture of the MRI scan", disabled=not enable)
    if st.button('Run image prediction'):
        img_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        result = img_model_prediction(img_bytes)
        # st.write(result)
        res_display = round(result[1].max()*100,2)
        # st.write(res_display)
        st.write('The model predicts the brain in the image is', mapping_img[result[0]], 'with a certainty of', res_display,'%')
        with st.sidebar:
            st.plotly_chart(px.pie(values=result[1].flatten(),names=mapping_img.values(), title='Probabilities of the prediction'),key=5) #,names=mapping.keys()

# pred = model.predict(user_input)
# Display the user input
# st.write(f'Hello potatoes, {st.session_state['result']}!')