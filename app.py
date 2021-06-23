import os
import sys
import pickle
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from util import base64_to_pil
from PIL import Image, ImageOps
import numpy as np
import gc


# Declare a flask app
app = Flask(__name__)



#Predict Functions
def import_and_predict(test):
	models = load_model('/content/drive/MyDrive/Plant Doctor/simple_classifier/Epoch-50-Val_acc1.00.hdf5')
	lbs=open(r'/content/drive/MyDrive/Plant Doctor/simple_classifier/label_transform.pkl', 'rb')
	k=predi(test,models,lbs)
	if k == 'Banana':
		model=load_model('/content/drive/MyDrive/Plant Doctor/banana_classifier/Epoch-48-Val_acc0.95.hdf5')
		lb=open(r'/content/drive/MyDrive/Plant Doctor/banana_classifier/label_transform.pkl', 'rb')
		pred=predic(test,lb,model)
		del model
		del lb
		gc.collect()
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
		model=load_model('/content/drive/MyDrive/Plant Doctor/paddy_classifier/Epoch-05-Val_acc0.82.hdf5')
		lb=open(r'/content/drive/MyDrive/Plant Doctor/paddy_classifier/label_transform.pkl', 'rb')
		pred=predic(test,lb,model)
		del model
		del lb
		gc.collect()
		if pred=='Paddy_LeafBlast':
			pred='Paddy Leaf Blast'
		elif pred=='Paddy_BrownSpot':
			pred='Paddy Brown Spot '
		elif pred=='Paddy_Healthy':
			pred='Paddy Healthy' 
		else:
			pred='Not a suitable leaf class'           
    
	elif k == 'Tomato':
		model=load_model('/content/drive/MyDrive/Plant Doctor/tomato_classifier/Epoch-29-Val_acc0.99.hdf5')
		lb=open(r'/content/drive/MyDrive/Plant Doctor/tomato_classifier/label_transform.pkl', 'rb')
		pred=predic(test,lb,model)
		del model
		del lb
		gc.collect()   
		if pred=='Tomato_Late_blight':
			pred='Tomato Late Blight'
		elif pred=='Tomato__Tomato_mosaic_virus':
			pred='Tomato Mosaic Virus'
		elif pred=='Tomato_healthy':
			pred='Tomato Healthy'
		else:
			pred='Not a suitable leaf class'

	else:
		pred='Not a suitable leaf class'
		
	return pred

def predic(test_image,labelencoder,mod):
    labeltransformer = pickle.load(labelencoder)
    test_image =image.img_to_array(test_image)
    test_image = np.array([test_image], dtype=np.float16) / 255.0
    prediction =mod.predict(test_image)
    print (np.max(prediction))
    del test_image
    del labelencoder
    gc.collect()
    if(np.max(prediction)>0.5):
       sim=labeltransformer.inverse_transform(prediction)
    else :
        sim="Not a suitable type"
    return sim

def predi(test_image,mod,labelencod):
    labeltransform = pickle.load(labelencod)
    test_image =image.img_to_array(test_image)
    test_image = np.array([test_image], dtype=np.float16) / 255.0
    prediction = mod.predict(test_image)
    print (np.max(prediction))
    del test_image
    del labelencod
    gc.collect()
    if(np.max(prediction)>0.999):
        sim= labeltransform.inverse_transform(prediction)
    else :
        sim="Not a suitable type"
    
    return  sim

 # Definition   
@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		try:
			size=(256,256)
			img = base64_to_pil(request.json)
			img = img.resize(size)
			result = import_and_predict(img)
			return jsonify(result=result)
		except ValueError:
			result = 'Image is not suitable. The image contains more than 3 input layers'
			return jsonify(result=result)
			
	return None

# Main function
if __name__ == '__main__':
	app.run()