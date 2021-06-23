import streamlit as st
import pickle
import numpy as np
import cv2
from keras.preprocessing import image
from PIL import Image, ImageOps
from keras.models import load_model 
import gc

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)


def import_and_predict(test_image):
    models = load_model('/home/nivethjunnithan/Desktop/PlantDoctor/simple_classifier/Epoch-50-Val_acc1.00.hdf5')
    lbs=open(r'/home/nivethjunnithan/Desktop/PlantDoctor/simple_classifier/label_transform.pkl', 'rb')
    labeltransformer = pickle.load(lbs)
    test_image = image.img_to_array(test_image)
    test_image = np.array([test_image], dtype=np.float16) / 255.0
    prediction = models.predict(test_image)
    del models
    del lbs
    gc.collect()
    
    if(np.max(prediction)>0.99):
        k=labeltransformer.inverse_transform(prediction)
    else :
        k="not a mentioned category"
		
    if k == 'Banana':
        model=load_model('/home/nivethjunnithan/Desktop/PlantDoctor/banana_classifier/Epoch-45-Val_acc0.99.hdf5')
        lb=open(r'/home/nivethjunnithan/Desktop/PlantDoctor/banana_classifier/label_transform.pkl', 'rb')
        pred=predic(test_image,lb,model)
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
        model=load_model('/home/nivethjunnithan/Desktop/PlantDoctor/paddy_classifier/Epoch-05-Val_acc0.82.hdf5')
        lb=open(r'/home/nivethjunnithan/Desktop/PlantDoctor/paddy_classifier/label_transform.pkl', 'rb')
        pred=predic(test_image,lb,model)
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
        return pred
    
    elif k == 'Tomato':
        model=load_model('/home/nivethjunnithan/Desktop/PlantDoctor/tomato_classifier/Epoch-29-Val_acc0.99.hdf5')
        lb=open(r'/home/nivethjunnithan/Desktop/PlantDoctor/tomato_classifier/label_transform.pkl', 'rb')
        pred=predic(test_image,lb,model)
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
        return pred

    else:
        pred='not a suitable leaf class'		
        return pred



def predic(test_image,labelencod,mod):
	labeltransform= pickle.load(labelencod)
	test_image =image.img_to_array(test_image)
	test_image = np.array([test_image], dtype=np.float16) / 255.0
	prediction =mod.predict(test_image)
	del test_image
	gc.collect()
	sim= labeltransform.inverse_transform(prediction)
	return sim


def about():
	st.write(
		'''
		**A convolutional neural network (CNN)** is a type of artificial neural network used in image recognition and processing that is specifically designed to process pixel data.
        A CNN uses a system much like a multilayer perceptron that has been designed for reduced processing requirements. The layers of a CNN consist of an input layer, an output layer and a hidden layer that includes multiple convolutional layers, pooling layers, fully connected layers and normalization layers. 
        
Read more :point_right: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

The base paper of reference for this project is:

https://drive.google.com/file/d/1tEOhwI91oEwY6gH36xP0jx_DAgBidiBU/view?usp=sharing
		''')

size = tuple((256, 256))

def main():
    st.title("Simple Classifier")
    st.write("**Banana Paddy Tomato**")
    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)
    
    if choice == "Home":
        st.write("Go to the About section from the sidebar to learn more about it.")
        file = st.file_uploader("Upload the diseased image", type=["PNG","JPG","JPEG"])
        
        if file is not None:
            im1 =Image.open(file)
            if st.button("Process"):
                st.image(im1)
                try:
                    predict= import_and_predict(im1)
                    string="The image is most likely to be "+predict
                    st.success (string)
                except ValueError:
                    st.success("Use another image")
    elif choice == "About":
    	about()
        
if __name__ == "__main__":
    main()
        
