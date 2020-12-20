# Realtime-Emotion-Detection-model-using-CNN
## DESCRIPTION:
Our Approach is to detect the facial emotions using the Facial Action Units.
### FACS(Facial action coding system):
Facial Muscular Movements causing momentary changes in the facial expressions are termed as
AU(Action Units)
Facial Expressions are identified by using a single Action unit or a series of action units which are
associated with facial movements.
## DATASET DESCRIPTION:
### Dataset:
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-
challenge/data
We are using 48x48 pixel grayscale images of faces
### Data Classified into -> seven categories:
1. 0=Angry
2. 1=Disgust
3. 2=Fear
4. 3=Happy
5. 4=Sad
6. 5=Surprise
7. 6=Neutral
Columns: "emotion" and "pixels"
No. of Classes: 7 (7 Emotions)
# TOOLS:
1. Keras
2. TensorFlow
3. OpenCV
4. Pandas
5. Numpy
6. Flask
7. Pillow
8. Gunicorn

Training Data: 28,709 samples.
Testing Data: 7178 samples

# PROJECT DIRCTORY: Real Time Emotion Detection from Facial Expression using CNN
Link: https://drive.google.com/drive/folders/161Yi5qqLs9jBsGJGNPY54eKlwmSE8KiJ
# Project Directory Structure:
## 0. Data:
--> test.csv
--> train.csv
1. Training Model Version 1:
--> Brightness_And_Sharpness_Augmented_data2.ipynb
--> CNN_Model.py
--> CNN_Model1.h5
--> CNN_Model_Epochs.csv
--> CNN_Model_Final_v1.h5
--> CNN_Model_Plot.png
--> Data_Preprocessing.py
--> Real_Time_Emotion_Detection_from_using_CNN_Version1.ipynb
--> Utils_funX.py
--> test.csv (Now in the Data Directory)
--> train.csv (Now in the Data Directory)
## 2. Realtime Webcam Prediction Version 1 (Deploying App on Localhost)
--> app.py
--> Real_Time_Webcam_Demonstration.ipynb
--> CNN_Model_Final_v1.h5
--> Real_Time_Webcam_Demonstration.py
--> Recorded_Time_Webcam_Demonstration.py
--> image_tools.py
--> Utils_funX.py
--> opencv-dnn
--> deploy.prototxt
--> weights.caffemodel
## 3. Realtime Webcam Prediction Version 2 (Deploying App on Heroku)
--> app.py
--> CNN_Model_Final_v1.h5
--> Real_Time_Webcam_Demonstration.py
--> requirements.txt
--> image_tools.py
--> Utils_funX.py
--> deploy.prototxt
--> weights.caffemodel
--> Aptfile
--> Procfile
--> runtime.txt
