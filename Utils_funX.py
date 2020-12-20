import numpy as np
    
def Decode_Y_Val(label):
    emotions =  ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
    class_val = np.argmax(label)
    return emotions[class_val]
    
