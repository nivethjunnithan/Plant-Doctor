import os
import sys
import pickle
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps


# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

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
    size=(256,256)
    labeltransformer = pickle.load(labelencoder)
    test_image =ImageOps.fit (test_image,size,Image.ANTIALIAS)
    test_image =image.img_to_array(test_image)
    test_image = np.array([test_image], dtype=np.float16) / 255.0
    prediction =mod.predict(test_image)
    sim= labeltransformer.inverse_transform(prediction)
    return sim
    
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        result = import_and_predict(img)

        # Serialize the result, you can add additional fields
        return jsonify(result=result)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
