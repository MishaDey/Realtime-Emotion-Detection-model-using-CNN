import cv2
import numpy as np
from tensorflow.keras import *
from tensorflow.keras.preprocessing import image
from Utils_funX import *
import image_tools as lit

prototxt_path = 'deploy.prototxt'
caffemodel_path ='weights.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path,caffemodel_path)

model = models.load_model('CNN_Model_Final_v1.h5')
#model.summary()


#facial_haarcascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
capture = cv2.VideoCapture(-1)

ex = 0
while(True):
    #Capturing the frame
    ret , facial_img = capture.read()
    if not ret:
        continue
        
        
    (h,w) = facial_img.shape[:2]
    gray_scale_img = np.array(lit.rgb2gray_approx(facial_img),dtype = 'uint8') 
    #gray_scale_img = cv2.cvtColor(facial_img,cv2.COLOR_BGR2GRAY)
    
    #faces = facial_haarcascade_classifier.detectMultiScale(gray_scale_img,scaleFactor=1.3,minNeighbors=5)
    
    blob = cv2.dnn.blobFromImage(cv2.resize(facial_img,(1000,1000)),1.0,(300,300),(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()      
    
    for i in range(0,detections.shape[2]):
        #Bounding Box
        confidence = detections[0,0,i,2]
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        
        
        (s_x,s_y,e_x,e_y) = box.astype("int")
        
        if(confidence > 0.19 ):
            #Draw the rectangle on the image
            cv2.rectangle( facial_img , (s_x,s_y) , (e_x,e_y) , color = (0,0,0) , thickness = 1 )
            #Cropping Out the Region of interest to feed to our trained             print(gray_cropped)
            gray_cropped = cv2.resize(gray_scale_img[s_y-20:e_y+20,s_x-10:e_x+10],(48,48))
        
            #getting the image pixels
            face_img_pixels1 = image.img_to_array(gray_cropped)
        
            face_img_pixels = np.expand_dims(face_img_pixels1,axis=0)
            face_img_pixels = face_img_pixels.astype(float)
        
            face_img_pixels /= 255 # Normalization        
        
            predicted_label = model.predict(face_img_pixels)
            predicted_emotion = Decode_Y_Val(predicted_label)
        
            #predicted_emotion = "I am Always Sad"
            predicted_emotion = predicted_emotion.upper()
            cv2.putText(facial_img,predicted_emotion,(int(s_x),int(s_y)-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)  
        
            Final_Result = cv2.resize(facial_img,(1920,1080))  
            cv2.imshow('Real Time Emotion Detection',Final_Result)
    
        if cv2.waitKey(10) == ord('q'):
            ex = 1
            break
            
    if ex == 1 :
        break
        
capture.release()
cv2.destroyAllWindows()
