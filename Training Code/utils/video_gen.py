import numpy as np
import tensorflow as tf
import cv2 as cv
import os 
import random
from os.path import join,exists
from glob import glob
from tensorflow.keras.utils import to_categorical
from natsort import natsorted


## Function to Sample the Frames from the Video

class FrameGenerator:

    def __init__(self,rgb_path,training=True):

       

        self.rgb_path=rgb_path
        
        self.training=training
        
        
        self.classnames=natsorted(os.listdir(self.rgb_path))
        self.num_class=len(self.classnames)
        

    
    def get_label(self,npz_file):
        classname=os.path.basename(npz_file).split('_')[0]
        return self.classnames.index(classname)

    def load_npy(self,npy_file):
        features=np.load(npy_file)
        return features

    
    def __call__(self):

        
        rgb_npy_list=natsorted(glob(self.rgb_path+'/**/*.npy'))

        
        

        if self.training:
            random.shuffle(rgb_npy_list)

        for rgb_npy_file in rgb_npy_list:
            
            rgb_imgs=self.load_npy(rgb_npy_file)

            
            label=self.get_label(rgb_npy_file)

            # feat_dict=self.combine_inputs(pose_feature,rgb_imgs,label)

            yield rgb_imgs,to_categorical(label,num_classes=self.num_class)