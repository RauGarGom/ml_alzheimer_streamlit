# import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.metrics import auc, roc_auc_score, accuracy_score, confusion_matrix
# from xgboost import XGBClassifier
import pickle
from tensorflow.keras import models
# from matplotlib.image import imread
import cv2

### Loading of models
with open('./model_2.pkl', 'rb') as f:
    class_model = pickle.load(f)
img_model = models.load_model("./baseline_model_b.keras")
img_scal = pickle.load(open("./aux_scal.pkl",'rb'))


def model_prediction(mmse="1",funct_asses=1,memory="Yes",behav="Yes",adl=1):
    ### A first part cleans the input given by Streamlit
    ### TODO: limpiar modelo, hay cosas que no hacen mucho
    mmse = int(mmse)
    memory = np.where(memory=='Yes',1,0)
    behav = np.where(behav=='Yes',1,0)

    result = class_model.predict(np.array([[mmse,funct_asses,memory,behav,adl]]))
    text_result = str(np.where(result == 1, "Patient presents signs of alzheimer. Please confirm with MRI prediction","Patient shows no signs of alzehimer")[0])
    result_proba = class_model.predict_proba(np.array([[mmse,funct_asses,memory,behav,adl]]))
    result_stream = result
    if (result_proba.max() < 0.9) and (result==0):
        text_result = str('Patient shows no signs of alzheimer, but the model is unsure. MRI model prediction is advised')
        result_stream = 2
    if  (result_proba.max() < 0.9) and (result==1):
        text_result = str('Patient shows signs of alzheimer, but the model is unsure. MRI model prediction is anyway advised')
        result_stream = 3
    return result, text_result, result_proba, result_stream

def img_model_prediction(image_path,img_size=32):
    '''Img_size must be the same as the one used by the training of the model.
    Model 4 (used in the demo) is made with 64x64'''
    image = cv2.imdecode(image_path, cv2.IMREAD_COLOR)
    image = recortar_centro_relativo(1,0.5)
    # mapping = {
    #     0: 'Non Demented',
    #     1: 'Very Mild Demented',
    #     2: 'Mild Demented',
    #     3: 'Moderate Demented'
    # }
    if img_size == 32:
        image = cv2.resize(image, (32, 32)) ### 32x32 pixels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) ### Conversion to gray scale
        image = image.reshape(-1,1)
        image = img_scal.transform(image)
        image = image.reshape(-1, 32, 32, 1)
    else:
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) ### Conversion to gray scale
        image = image.reshape(-1,1)
        image = img_scal.transform(image)      
        image = image.reshape(-1, 64, 64, 1)        
    img_pred = img_model.predict(image)
    return img_pred.argmax(),img_pred.round(4)

def recortar_centro_relativo(imagen, porcentaje_ancho=1, porcentaje_alto=0.5):
    alto, ancho, _ = imagen.shape  # Dimensiones de la imagen
    
    # Calcular dimensiones del recorte
    ancho_corte = int(ancho * porcentaje_ancho)
    alto_corte = int(alto * porcentaje_alto)
    
    # Coordenadas centrales
    centro_x, centro_y = ancho // 2, alto // 2
    
    # Coordenadas del recorte
    x_inicio = max(centro_x - ancho_corte // 2, 0)
    x_fin = min(centro_x + ancho_corte // 2, ancho)
    y_inicio = max(centro_y - alto_corte // 2, 0)
    y_fin = min(centro_y + alto_corte // 2, alto)
    
    # Recortar la imagen
    recorte = imagen[y_inicio:y_fin, x_inicio:x_fin]
    return recorte