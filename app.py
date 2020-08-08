import numpy as np
import sys
import glob

import os
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model
from keras import backend
from tensorflow.keras import backend
from keras.preprocessing import image
import tensorflow as tf

global graph

sess = tf.Session()
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.applications.imagenet_utils import preprocess_input,decode_predictions

app = Flask(__name__)
set_session(sess)
model = load_model("rock.h5")
print("model loaded")


@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        
        basepath = os.path.dirname(__file__)
        
        filepath = os.path.join(basepath,'uploads',secure_filename(f.filename))
        
        f.save(filepath)
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        global sess
        with graph.as_default():
            set_session(sess)
            preds = model.predict(x)
            preds1=np.ravel(preds)
            pred = np.argmax(preds1)
            print("pred",pred)
            
            
            
            print("prediction",preds)
            
        index = ['bornite','chrysocolla','malachite','muscovite','pyrite','quartz']
        
        text = "The Predicted Rock is  " + index[pred]
        
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    