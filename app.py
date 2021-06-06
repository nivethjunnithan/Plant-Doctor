import os
import sys
import pickle
import tensorflow as tf
import numpy as np
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from util import base64_to_pil

# Some utilites



# Declare a flask app
app = Flask(__name__)


#Predict Functions
def import_and_predict(test):
    models = load_model('/home/nivethjunnithan/Desktop/PlantDoctor/simple_classifier/Epoch-50-Val_acc1.00.hdf5')
    lbs=open(r'/home/nivethjunnithan/Desktop/PlantDoctor/simple_classifier/label_transform.pkl', 'rb')
    k=predic(test,lbs,models)
    lbs.close()
    if k == 'Banana':
        model=load_model('/home/nivethjunnithan/Desktop/PlantDoctor/banana_classifier/Epoch-45-Val_acc0.99.hdf5')
        lb=open(r'/home/nivethjunnithan/Desktop/PlantDoctor/banana_classifier/label_transform.pkl', 'rb')
        pred=predic(test,lb,model)
        lb.close() 
        if pred=='Banana_bbs':
            pred='Black Sigatoka of Banana'
        elif pred=='Banana_bbw':
            pred='Xanthomonas Wilt of Banana'
        elif pred=='Banana_Healthy':
            pred='Banana Healthy'
        else:
            pred='Not a suitable leaf class'
        return pred
        
    elif k == 'Paddy':
        model=load_model('/home/nivethjunnithan/Desktop/PlantDoctor/paddy_classifier/Epoch-05-Val_acc0.82.hdf5')
        lb=open(r'/home/nivethjunnithan/Desktop/PlantDoctor/paddy_classifier/label_transform.pkl', 'rb')
        pred=predic(test,lb,model)
        lb.close() 
        if pred=='Paddy_LeafBlast':
            pred='Paddy Leaf Blast'
        elif pred=='Paddy_BrownSpot':
            pred='Paddy Brown Spot '
        elif pred=='Paddy_Healthy':
            pred='Paddy Healthy' 
        else:
            pred='Not a suitable leaf class'           
        return pred
    
    elif k == 'Tomato':
        model=load_model('/home/nivethjunnithan/Desktop/PlantDoctor/tomato_classifier/Epoch-29-Val_acc0.99.hdf5')
        lb=open(r'/home/nivethjunnithan/Desktop/PlantDoctor/tomato_classifier/label_transform.pkl', 'rb')
        pred=predic(test,lb,model)   
        lb.close()    
        if pred=='Tomato_Late_blight':
            pred='Tomato Late Blight'
        elif pred=='Tomato__Tomato_mosaic_virus':
            pred='Tomato Mosaic Virus'
        elif pred=='Tomato_healthy':
            pred='Tomato Healthy'
        else:
            pred='Not a suitable leaf class'
        return pred
        
    else:
        pred='Not a suitable leaf class'



def predic(test_image,labelencoder,mod):
    labeltransformer = pickle.load(labelencoder)
    test_image =image.img_to_array(test_image)
    test_image = np.array([test_image], dtype=np.float16) / 255.0
    prediction =mod.predict(test_image)
    sim= labeltransformer.inverse_transform(prediction)
    return sim

    
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        result = import_and_predict(img)
        return jsonify(result=result)

    return None

#Main function
if __name__ == '__main__':
	app.run()

