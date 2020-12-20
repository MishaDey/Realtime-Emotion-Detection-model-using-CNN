import io
import base64
from PIL import Image
from io import StringIO
import cv2
import numpy as np
import tensorflow
from tensorflow.keras import *
from Utils_funX import *
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin

prototxt_path = 'deploy.prototxt'
caffemodel_path ='weights.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path,caffemodel_path)

app = Flask(__name__)
socketio = SocketIO(app)

def getModel():
    global classifier
    classifier = tensorflow.keras.models.load_model('CNN_Model_Final_v1.h5')
getModel()

cors = CORS(app)
#app.config['CORS_HEADERS'] = 'Content-Type'
@app.route('/', methods=['POST', 'GET'])
@cross_origin()

def index():
    return render_template('index.html')

@socketio.on('image')
def image(data_image):
    sbuf = StringIO()
    sbuf.write(data_image)
    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    facial_img = Image.open(b)

    (h,w) = np.float32(facial_img).shape[:2]

    gray_scale_img = np.array(cv2.cvtColor(np.float32(facial_img), cv2.COLOR_BGR2GRAY),dtype = 'uint8')

    blob = cv2.dnn.blobFromImage(cv2.resize(np.float32(facial_img),(1000,1000)),1.0,(300,300),(104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()

    for i in range(0,detections.shape[2]):

        confidence = detections[0,0,i,2]
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        (s_x,s_y,e_x,e_y) = box.astype("int")
        
        if(confidence > 0.19 ):
            gray_cropped = cv2.resize(gray_scale_img[s_y-20:e_y+20,s_x-10:e_x+10],(48,48))
        
            face_img_pixels1 = tensorflow.keras.preprocessing.image.img_to_array(gray_cropped)
        
            face_img_pixels = np.expand_dims(face_img_pixels1,axis=0)
            face_img_pixels = face_img_pixels.astype(float)
        
            face_img_pixels /= 255       
        
            predicted_label = classifier.predict(face_img_pixels)
            predicted_emotion = Decode_Y_Val(predicted_label)
        
            predicted_emotion = predicted_emotion.upper()

            emit('response_back',predicted_emotion)
           
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0',debug=True)
