import numpy as np
from sklearn.preprocessing import normalize
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
from Utils_funX import *

def Data_Preparation(train_data,test_data):
    x_train,y_train,x_test,y_test=[],[],[],[]
    # pixel values are in x_values and the categories/clases are the y_values
    for index,row in train_data.iterrows():
        row_val=np.array(row['pixels'].split(" "))
        x_train.append(np.array(row_val,'float64'))
    x_train = np.array(x_train,'float64')
    y_train = train_data["emotion"].values
    
    for index,row in test_data.iterrows():
        row_val=np.array(row['pixels'].split(" "))
        x_test.append(np.array(row_val,'float64'))
    x_test = np.array(x_test,'float64')
    y_test=np.array(y_test)
    
    #y_test=y_test.reshape((y_test.shape[0],1))
    #y_train=y_train.reshape((y_train.shape[0],1))
    return x_train,y_train,x_test,y_test

def Data_Augmentation(x_train,y_train):
    # mirroring the train data() along the vertical and horozontal Axis
    # Concept --> Only the x_train will be changed --> that is the pixel values will be mirrored
    # the y_train values will remain same --> since the labels coressponding to each emotion will remail same
    for flip_type in ['vertical']:
        augmented_x_train = []
        augmented_y_train = []
        
        for i,image in enumerate(x_train):
            augmented_image,augmented_labels = get_mirrored_data(image,y_train[i],flip_type)
            augmented_x_train.append(augmented_image)
            augmented_y_train.append(augmented_labels)
        
        print("\nFive Sample of image after {} flip :\n".format(flip_type))
        for i in range(5):
            plt.imshow(augmented_x_train[i])
            plt.title(Decode_Y_Val(augmented_y_train[i]))
            plt.show()
                                     
        x_train = np.concatenate((x_train,augmented_x_train))
        y_train = np.concatenate((y_train,augmented_y_train))
        
    return x_train,y_train
    
def get_mirrored_data(image_pixel,label,flip_type):
    aug_img = pcv.flip(image_pixel,flip_type)
    aug_label = label 
    return aug_img,aug_label
    

def Data_Normalization(x_train,x_test):
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    return x_train,x_test