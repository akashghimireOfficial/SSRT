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

tf.random.set_seed(123)
np.random.seed(123)
random.seed(123)


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


class FrameGenerator_CD:

    def __init__(self,rgb_path,training=True):

       

        self.rgb_path=rgb_path
        
        self.training=training
        
        self.class_dict={'A004':'Drink.Fromcup','A032':'Readbook','A035':'Usetelephone','A037':'Uselaptop'}
        self.classnames=natsorted(os.listdir(self.rgb_path))
        self.num_class=len(self.classnames)
        

    
    

    def load_npy(self,npy_file):
        features=np.load(npy_file)
        return features

    def get_label(self,npz_file):
        # classname=os.path.basename(npz_file).split('_')[0]
        classname=self.class_dict[os.path.basename(npz_file).split('_')[0]]
        return self.classnames.index(classname)
    
    def __call__(self):

        
        rgb_npy_list=natsorted(glob(self.rgb_path+'/**/*.npy'))

        
        

        if self.training:
            random.shuffle(rgb_npy_list)

        for rgb_npy_file in rgb_npy_list:
            
            rgb_imgs=self.load_npy(rgb_npy_file)

            
            label=self.get_label(rgb_npy_file)

            # feat_dict=self.combine_inputs(pose_feature,rgb_imgs,label)

            yield rgb_imgs,to_categorical(label,num_classes=self.num_class)




class PoseGenerator:

    def __init__(self,pose_path,training=True):

       

        self.pose_path=pose_path
        
        self.training=training
        
        
        self.classnames=natsorted(os.listdir(self.pose_path))
        self.num_class=len(self.classnames)
        

    
    def get_label(self,npz_file):
        classname=os.path.basename(npz_file).split('_')[0]
        return self.classnames.index(classname)

    def load_npy(self,npy_file):
        features=np.load(npy_file)
        return features

    
    def __call__(self):

        
        pose_npy_list=natsorted(glob(self.pose_path+'/**/*.npy'))

        
        

        if self.training:
            random.shuffle(pose_npy_list)

        for pose_npy_file in pose_npy_list:
            
            poses=self.load_npy(pose_npy_file)

            
            label=self.get_label(pose_npy_file)

            # feat_dict=self.combine_inputs(pose_feature,rgb_imgs,label)

            yield poses,to_categorical(label,num_classes=self.num_class)



class PoseGenerator_CD:

    def __init__(self,pose_path,training=True):

       

        self.pose_path=pose_path
        
        self.training=training
        
        self.class_dict={'A004':'Drink.Fromcup','A032':'Readbook','A035':'Usetelephone','A037':'Uselaptop'}
        self.classnames=natsorted(os.listdir(self.pose_path))
        self.num_class=len(self.classnames)
        

    
    def get_label(self,npz_file):
        # classname=os.path.basename(npz_file).split('_')[0]
        classname=self.class_dict[os.path.basename(npz_file).split('_')[0]]
        return self.classnames.index(classname)

    def load_npy(self,npy_file):
        features=np.load(npy_file)
        return features

    
    def __call__(self):

        
        pose_npy_list=natsorted(glob(self.pose_path+'/**/*.npy'))

        
        

        if self.training:
            random.shuffle(pose_npy_list)

        for pose_npy_file in pose_npy_list:
            
            poses=self.load_npy(pose_npy_file)

            
            label=self.get_label(pose_npy_file)

            # feat_dict=self.combine_inputs(pose_feature,rgb_imgs,label)

            yield poses,to_categorical(label,num_classes=self.num_class)








class Fusiongenerator:

    def __init__(self,pose_path,rgb_path,training=True):

        self.pose_path=pose_path

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

        pose_npy_list=natsorted(glob(self.pose_path+'/**/*.npy'))

        # add_file=list(map(lambda x:'/'.join(x.split('/')[-2:]),pose_npy_list))

        
        # rgb_npy_list=list(map(lambda x:join(self.rgb_path,x),add_file))


        rgb_npy_list=natsorted(glob(self.rgb_path+'/**/*.npy'))

        combined_inputs=list(zip(pose_npy_list,rgb_npy_list))
        

        if self.training:
            random.shuffle(combined_inputs)

        for pose_npy_file,rgb_npy_file in combined_inputs:
            
            poses=self.load_npy(pose_npy_file)
            rgb_imgs=self.load_npy(rgb_npy_file)

            # print(pose_npy_file)
            # print(rgb_npy_file)

            
            label=self.get_label(rgb_npy_file)

            # feat_dict=self.combine_inputs(pose_feature,rgb_imgs,label)

            #yield pose_npy_file,rgb_npy_file,to_categorical(label,num_classes=self.num_class)


            yield poses,rgb_imgs,to_categorical(label,num_classes=self.num_class),to_categorical(label,num_classes=self.num_class)



class Fusiongenerator_CD:

    def __init__(self,pose_path,rgb_path,training=True):

        self.pose_path=pose_path

        self.rgb_path=rgb_path
        
        self.training=training
        
        self.class_dict={'A004':'Drink.Fromcup','A032':'Readbook','A035':'Usetelephone','A037':'Uselaptop'}
        self.classnames=natsorted(os.listdir(self.rgb_path))
        self.num_class=len(self.classnames)
        

    
    # def get_label(self,npz_file):
    #     classname=os.path.basename(npz_file).split('_')[0]
    #     return self.classnames.index(classname)

    def get_label(self,npz_file):
        # classname=os.path.basename(npz_file).split('_')[0]
        classname=self.class_dict[os.path.basename(npz_file).split('_')[0]]
        return self.classnames.index(classname)

    def load_npy(self,npy_file):
        features=np.load(npy_file)
        return features

    
    def __call__(self):

        pose_npy_list=natsorted(glob(self.pose_path+'/**/*.npy'))

        # add_file=list(map(lambda x:'/'.join(x.split('/')[-2:]),pose_npy_list))

        
        # rgb_npy_list=list(map(lambda x:join(self.rgb_path,x),add_file))


        rgb_npy_list=natsorted(glob(self.rgb_path+'/**/*.npy'))

        combined_inputs=list(zip(pose_npy_list,rgb_npy_list))
        

        if self.training:
            random.shuffle(combined_inputs)

        for pose_npy_file,rgb_npy_file in combined_inputs:
            
            poses=self.load_npy(pose_npy_file)
            rgb_imgs=self.load_npy(rgb_npy_file)

            # print(pose_npy_file)
            # print(rgb_npy_file)

            
            label=self.get_label(rgb_npy_file)

            # feat_dict=self.combine_inputs(pose_feature,rgb_imgs,label)

            #yield pose_npy_file,rgb_npy_file,to_categorical(label,num_classes=self.num_class)


            yield poses,rgb_imgs,to_categorical(label,num_classes=self.num_class),to_categorical(label,num_classes=self.num_class)

           