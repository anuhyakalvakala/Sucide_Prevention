#Import necessary libraries
import tokenize
from flask import Flask, render_template, redirect, flash, request, send_from_directory
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from werkzeug.utils import secure_filename
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from skimage import color
# from skimage import io
from PIL import Image


UPLOADS_FOLDER = '/Users/anuhyakalvakala/Desktop/sc/static/images'
# Create flask instance
app = Flask(__name__)

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOADS_FOLDER'] = UPLOADS_FOLDER

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph=tf.compat.v1.get_default_graph()

# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)

# Function to load and prepare the image in right shape
def read_image(filename):
    # Load the image
    # img = io.imread(filename)[:,:,:3]
    #img = tf.keras.utils.load_img(filename, grayscale=True)#, target_size=(48, 48))
    # img = color.rgb2gray(img)
    

    # img = Image.open(filename)
    #img = tf.keras.utils.load_img(filename,  target_size=(48, 48))#grayscale=True,
    # img = img.convert('L')
    # print(img.show())
    # print(io.imshow(img))
    # Convert the image to array
    img = tf.keras.utils.load_img(filename, grayscale=True, target_size=(48,48))

    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # img = image.img_to_array(img)
    # img = np.expand_dims(img, axis=0)
    # # Reshape the image into a sample of 1 channel
    # img = img.reshape(1, 48, 48, 1)
    # Prepare it as pixel data
    # img = img.astype('float32')
    # img = img / 255.0
    return img

@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('index.html')

@app.route("/predict", methods = ['GET','POST'])

def predict():
    if request.method == 'POST':
        file = request.files['file']
        texts = request.form['text']
        print("File Object", file)
        print("cmnt",texts)
        # try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename,'filename')
            basedir = os.path.abspath(os.path.dirname(__file__))
            print(basedir,'basedir')
            file_path = os.path.join(basedir, app.config['UPLOADS_FOLDER'], filename)
            print(os.path.join(basedir, app.config['UPLOADS_FOLDER'], filename))
            print(file_path,'file_path')
            file.save(file_path)
            img = read_image(file_path)
            print(img,'img')
            # Predict the class of an image

        with graph.as_default():
            label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}
            model1 = load_model('/Users/anuhyakalvakala/Downloads/model_cnn_td_2_new.h5')
            class_prediction = np.argmax(model1.predict(img))
            print(class_prediction)
            product = label_dict[class_prediction]
            print('The predicted emotion is : ' + label_dict[class_prediction])
            #predict comment
            model2 = load_model('/Users/anuhyakalvakala/Desktop/sc/comment_model.h5')
            tokenizer = Tokenizer()
            cmnt = tokenizer.texts_to_sequences(texts)
            cmnt = pad_sequences(cmnt, maxlen=203, dtype='int32', value=0)
            print(cmnt)
            class_prediction1 = np.argmax(model2.predict(cmnt,batch_size=2,verbose = 2)[0])
            print(class_prediction1,"class_prediction1-comment")

        if(class_prediction1 == 0):
            emo = "Negative"
        elif (class_prediction1 == 1):
            emo = "Positive" 
        else:
            emo = "Neutral"
        return render_template('predict.html', product = product, user_image =filename ,emo = emo,txt = texts)
        # except Exception as e:
        #     return "Unable to read the file. Please check if the file extension is correct."

    return render_template('predict.html')

if __name__ == "__main__":
    init()
    app.run(debug=True ,port=8080,use_reloader=False)