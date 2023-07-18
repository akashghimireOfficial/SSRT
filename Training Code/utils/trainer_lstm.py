# GENERAL LIBRARIES 
import math
import numpy as np
import joblib
from pathlib import Path
import pickle
# MACHINE LEARNING LIBRARIES
import sklearn
from sklearn.metrics import classification_report,balanced_accuracy_score
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import TimeDistributed
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# OPTUNA
import optuna
from optuna.trial import TrialState
# CUSTOM LIBRARIES

from utils.generator import FrameGenerator,PoseGenerator,Fusiongenerator
from utils.lstm_model import lstm,lstm_fusion
from utils.resnet183d import resnet3d
from utils.densenet import features_extraction_model
from utils.transformer import TransformerEncoder, pos_encoding,MultiHeadAttention,TransformerDecoder,Transformerfusion,Transformerfusion2
from utils.transformer_fus import Transformerfusion3
from utils.data import  random_flip, random_noise, one_hot,aug,combine_inputs

from utils.tools import CustomSchedule, CosineSchedule
from utils.tools import Logger

import copy
import random
import os
from glob import glob
from os.path import join,exists

tf.random.set_seed(123)
np.random.seed(123)
random.seed(123)

# TRAINER CLASS 
class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.split = 1
        self.fold = 0
        self.trial = None
        self.dataset=config['dataset']
        self.features=config['features']
        self.evaluation=config['evalutaion']
        self.num_class=self.config[self.evaluation]['CLASSES']
        self.result_dire=config['RESULTS_DIR']
        if self.features=='skeleton':
            self.pose=config[self.features]['pose']
            
            
            self.dataset_dire_pose=join('features/',self.dataset,self.features,self.pose,self.evaluation)

            self.shape=(32,34)

        elif self.features=='rgb':
            self.backbone=self.config[self.features]['backbone']
            self.dataset_dire_rgb=join('features/',self.dataset,self.features,self.backbone,self.evaluation)
            self.shape=(32,350,350,3)

        else:
            self.pose=config[self.features]['pose']
            
            
            
            self.dataset_dire_pose=join('features/',self.dataset,'skeleton',self.pose,self.evaluation)

            self.shape_pose=(32,34)

            self.backbone=self.config[self.features]['backbone']
            self.dataset_dire_rgb=join('features/',self.dataset,'rgb',self.backbone,self.evaluation,'crop')
            self.shape_rgb=(32,350,350,3)

            self.arch=config[self.features]['arch']

            

            

            
        self.result_dire=self.config['Result_dire']    




        self.bin_path = self.config['MODEL_DIR']
        
        self.model_size = self.config['MODEL_SIZE']
        self.n_heads = self.config[self.model_size]['N_HEADS']
        #self.n_heads = 256
        self.n_layers = self.config[self.model_size]['N_LAYERS']
        self.embed_dim = self.config[self.model_size]['EMBED_DIM']
        self.dropout = self.config[self.model_size]['DROPOUT']
        self.mlp_head_size = self.config[self.model_size]['MLP']
        self.activation = tf.nn.gelu
        self.d_model = 1 * self.n_heads
        
        self.d_ff = self.d_model * 2 



    def build_act(self,lstm_model):
        inputs = tf.keras.layers.Input(shape=self.shape)
        

        

        

        # x=tf.keras.layers.LSTM(units=self.d_model,return_sequences=True)(inputs)
        
        # x=tf.keras.layers.LayerNormalization()(x)

        
        

    
        
        x = lstm_model(inputs)



        
       
        outputs = tf.keras.layers.Dense(self.num_class,activation='softmax')(x)
        return tf.keras.models.Model(inputs, outputs)
        

    def build_fusion(self,lstm_fusion):
        input_pose = tf.keras.layers.Input(shape=self.shape_pose)
        input_rgb=  tf.keras.layers.Input(shape=self.shape_rgb)
        

        

        

        # x=tf.keras.layers.LSTM(units=self.d_model,return_sequences=True)(inputs)
        
        # x=tf.keras.layers.LayerNormalization()(x)

        
        

    
        
        x = lstm_fusion([input_pose,input_rgb],mask=None,training=True)

        

       
        
        
        outputs = tf.keras.layers.Dense(self.num_class)(x)
        return tf.keras.models.Model([input_pose,input_rgb], outputs)


    def build_rgb_BERT(self,resnet,lstm_model):
        

        inputs=tf.keras.Input(shape=(32,350,350,3))

        x=resnet(inputs)

        x = lstm_model(x,mask=None,training=True)


        
        
        #x = tf.keras.layers.Dense(self.mlp_head_size)(x)
        
        outputs = tf.keras.layers.Dense(self.num_class)(x)
        return tf.keras.models.Model(inputs, outputs)



    


    def get_model(self,trial_num):

       
        

        if self.features=='skeleton':
            lstm_m=lstm()
            self.model = self.build_act(lstm_m)
            
        elif self.features=='rgb':
            lstm_m = lstm()
            if self.backbone=='raw':
                resnet=resnet3d()
                
                self.model=self.build_rgb_BERT(resnet,lstm_m)
            

           


        else:
            #transformer = TransformerEncoder(d_model=self.d_model,dff=self.d_ff,num_heads=self.n_heads,rate=self.dropout,num_layers=self.n_layers)
            transformer_fusion=lstm_fusion()
            self.model=self.build_fusion(lstm_fusion)


        
            
        
        



        # lr = CustomSchedule(self.d_model, 
        #      warmup_steps=1190*self.config['N_EPOCHS']*self.config['WARMUP_PERC'],
        #      decay_step=1190*self.config['N_EPOCHS']*self.config['STEP_PERC'])



        optimizer = tf.keras.optimizers.Adam()

        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'],
                           )

        self.name_model_bin = self.config['MODEL_NAME'] + '_'  + '_' + str(self.split) + '_' + str(self.fold)
        
        if self.features=='skeleton':

            self.new_bin_path=join(self.bin_path,self.dataset,self.features,self.pose,self.evaluation,'{:04d}/'.format(trial_num))
        elif self.features=='rgb':
            self.new_bin_path=join(self.bin_path,self.dataset,self.features,self.backbone,self.evaluation,'{:04d}/'.format(trial_num))

        else:

            self.new_bin_path=join(self.bin_path,self.dataset,self.features,self.arch,self.evaluation,'{:04d}/'.format(trial_num))


        

        


        self.checkpointer = tf.keras.callbacks.ModelCheckpoint(self.new_bin_path+self.name_model_bin,
                                                               monitor="val_accuracy",
                                                               save_best_only=True,
                                                               save_weights_only=True)

    def get_data_pose(self):

        
        
        batch_size=self.config['BATCH_SIZE']
        train_dire=join(self.dataset_dire_pose,'train')
        val_dire=join(self.dataset_dire_pose,'val')
        test_dire=join(self.dataset_dire_pose,'test')

    	
        
        output_signature = (tf.TensorSpec(shape = (None, 34), dtype = tf.float32),
                    tf.TensorSpec(shape = (3), dtype = tf.int16))
        
        


        self.ds_train_pose=tf.data.Dataset.from_generator(PoseGenerator(pose_path=train_dire,training=True),output_signature=output_signature)
        self.ds_train_pose=self.ds_train_pose.cache('cpose/train/')
        self.ds_train_pose=self.ds_train_pose.batch(batch_size)
        self.ds_train_pose=self.ds_train_pose.prefetch(tf.data.experimental.AUTOTUNE)

        self.ds_test_pose=tf.data.Dataset.from_generator(PoseGenerator(pose_path=test_dire,training=True),output_signature=output_signature)
        self.ds_test_pose=self.ds_test_pose.cache('cpose/test/')
        self.ds_test_pose=self.ds_test_pose.batch(batch_size)
        self.ds_test_pose=self.ds_test_pose.prefetch(tf.data.experimental.AUTOTUNE)

        self.ds_val_pose=tf.data.Dataset.from_generator(PoseGenerator(pose_path=val_dire,training=True),output_signature=output_signature)
        self.ds_val_pose=self.ds_val_pose.cache('cpose/val/')
        self.ds_val_pose=self.ds_val_pose.batch(batch_size)
        self.ds_val_pose=self.ds_val_pose.prefetch(tf.data.experimental.AUTOTUNE)



    def get_data_rgb(self):

        batch_size=self.config['BATCH_SIZE']
        train_dire=join(self.dataset_dire_rgb,'train')
        val_dire=join(self.dataset_dire_rgb,'val')
        test_dire=join(self.dataset_dire_rgb,'test')

        output_signature = (tf.TensorSpec(shape = (None,None,None, 3), dtype = tf.float32),
                tf.TensorSpec(shape = (3), dtype = tf.int16))
    

        self.ds_train_rgb=tf.data.Dataset.from_generator(FrameGenerator(rgb_path=train_dire,training=True),output_signature=output_signature)
        self.ds_train_rgb=self.ds_train_rgb.cache('crgb/train/')
        self.ds_train_rgb=self.ds_train_rgb.batch(batch_size)
        self.ds_train_rgb=self.ds_train_rgb.prefetch(tf.data.experimental.AUTOTUNE)

        self.ds_test_rgb=tf.data.Dataset.from_generator(FrameGenerator(rgb_path=test_dire,training=True),output_signature=output_signature)
        self.ds_test_rgb=self.ds_test_rgb.cache('crgb/test/')
        self.ds_test_rgb=self.ds_test_rgb.batch(batch_size)
        self.ds_test_rgb=self.ds_test_rgb.prefetch(tf.data.experimental.AUTOTUNE)

        self.ds_val_rgb=tf.data.Dataset.from_generator(FrameGenerator(rgb_path=val_dire,training=True),output_signature=output_signature)
        self.ds_val_rgb=self.ds_val_rgb.cache('crgb/val/')
        self.ds_val_rgb=self.ds_val_rgb.batch(batch_size)
        self.ds_val_rgb=self.ds_val_rgb.prefetch(tf.data.experimental.AUTOTUNE)





        

       


    

    def get_data_fusion(self):
        
        batch_size=self.config['BATCH_SIZE']
        train_dire_pose=join(self.dataset_dire_pose,'train')
        val_dire_pose=join(self.dataset_dire_pose,'val')
        test_dire_pose=join(self.dataset_dire_pose,'test')

        train_dire_rgb=join(self.dataset_dire_rgb,'train')
        val_dire_rgb=join(self.dataset_dire_rgb,'val')
        test_dire_rgb=join(self.dataset_dire_rgb,'test')

        output_signature = (tf.TensorSpec(shape = (None,34), dtype = tf.float32),
                        tf.TensorSpec(shape = (None,None,None,3), dtype = tf.float32),
                        tf.TensorSpec(shape = (3), dtype = tf.int16))

        
        self.ds_train_fusion=tf.data.Dataset.from_generator(Fusiongenerator(pose_path=train_dire_pose,rgb_path=train_dire_rgb,training=True),output_signature=output_signature)
        self.ds_train_fusion=self.ds_train_fusion.map(combine_inputs)
        self.ds_train_fusion=self.ds_train_fusion.cache('cfusion/train/')
        self.ds_train_fusion=self.ds_train_fusion.batch(batch_size)
        self.ds_train_fusion=self.ds_train_fusion.prefetch(tf.data.experimental.AUTOTUNE)


        self.ds_test_fusion=tf.data.Dataset.from_generator(Fusiongenerator(pose_path=test_dire_pose,rgb_path=test_dire_rgb,training=False),output_signature=output_signature)
        self.ds_test_fusion=self.ds_test_fusion.map(combine_inputs)
        self.ds_test_fusion=self.ds_test_fusion.cache('cfusion/test/')
        self.ds_test_fusion=self.ds_test_fusion.batch(batch_size)
        self.ds_test_fusion=self.ds_test_fusion.prefetch(tf.data.experimental.AUTOTUNE)


        self.ds_val_fusion=tf.data.Dataset.from_generator(Fusiongenerator(pose_path=val_dire_pose,rgb_path=val_dire_rgb,training=False),output_signature=output_signature)
        self.ds_val_fusion=self.ds_val_fusion.map(combine_inputs)
        self.ds_val_fusion=self.ds_val_fusion.cache('cfusion/val/')
        self.ds_val_fusion=self.ds_val_fusion.batch(batch_size)
        self.ds_val_fusion=self.ds_val_fusion.prefetch(tf.data.experimental.AUTOTUNE)












    






        



    


        
        

        

        
        
    def get_random_hp(self):
        
        
        self.logger.save_log('\nRN_STD: {:.2e}'.format(self.config['RN_STD']))
        self.logger.save_log('EPOCHS: {}'.format(self.config['N_EPOCHS']))
        self.logger.save_log('WARMUP_PERC: {:.2e}'.format(self.config['WARMUP_PERC']))
        self.logger.save_log('WEIGHT_DECAY: {:.2e}\n'.format(self.config['WEIGHT_DECAY']))
        
    def do_training(self,trial_num=0):


        if self.features=='skeleton':
            self.get_data_pose()
            self.get_model(trial_num)
            history = self.model.fit(self.ds_train_pose,
                epochs=self.config['N_EPOCHS'], initial_epoch=0,
                validation_data=self.ds_val_pose,
                callbacks=[self.checkpointer], verbose=self.config['VERBOSE'])

           
                
            _, accuracy_test = self.model.evaluate(self.ds_test_pose)

            
            X, y_true_cat = tuple(zip(*self.ds_test_pose))
            y_true_cat=tf.concat(y_true_cat, axis=0)
            y_pred_cat = tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1)
            y_true=np.argmax(y_true_cat,axis=1)
            y_pred=np.argmax(y_pred_cat,axis=1)
            
            
            
            
            text = f"{self.config['MODEL_SIZE']}:  Accuracy Test: {accuracy_test} \n"

            
            
            
            
            self.logger.save_log(text)

            #print('len :',len(self.ds_train))
        elif self.features=='rgb':
            if self.backbone=='raw':
                self.get_data_rgb()

            

            self.get_model(trial_num)

            history = self.model.fit(self.ds_train_rgb,
                epochs=self.config['N_EPOCHS'], initial_epoch=0,
                validation_data=self.ds_val_rgb,
                callbacks=[self.checkpointer], verbose=self.config['VERBOSE'])

            _, accuracy_test = self.model.evaluate(self.ds_test_rgb)
            X, y_true_cat = tuple(zip(*self.ds_test_rgb))
            y_true_cat=tf.concat(y_true_cat, axis=0)
            y_pred_cat = tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1)
            y_true=np.argmax(y_true_cat,axis=1)
            y_pred=np.argmax(y_pred_cat,axis=1)
            # print('Shape: ',y_true.shape)
            
            
            
            
            # class_report=classification_report(y_true,y_pred)


            # balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true,y_pred)

            text = f"{self.config['MODEL_SIZE']}:  Accuracy Test: {accuracy_test} \n"

            
            
            # text1 = 'Classification Report \n{}'.format(class_report)
            
            self.logger.save_log(text)
            # self.logger.save_log(text1)

            

        else:
            self.get_data_fusion()

            self.get_model(trial_num)

            history = self.model.fit(self.ds_train_fusion,
                epochs=self.config['N_EPOCHS'], initial_epoch=0,
                validation_data=self.ds_val_fusion,
                callbacks=[self.checkpointer], verbose=self.config['VERBOSE'])

            
            
                
            _, accuracy_test = self.model.evaluate(self.ds_test_fusion)

            X, y_true_cat = tuple(zip(*self.ds_test_fusion))
            y_true_cat=tf.concat(y_true_cat, axis=0)
            y_pred_cat = tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1)
            y_true=np.argmax(y_true_cat,axis=1)
            y_pred=np.argmax(y_pred_cat,axis=1)
            
            
            # text1 = 'Classification Report \n{}'.format(class_report)
            
            self.logger.save_log(text)

 
            

        
        if self.features=='skeleton':

            pkl_folder=join(self.result_dire,self.dataset,self.features,self.pose,self.evaluation,self.config['MODEL_SIZE'])
        elif self.features=='rgb':
            pkl_folder=join(self.result_dire,self.dataset,self.features,self.backbone,self.evaluation,self.config['MODEL_SIZE'])

        else:

            pkl_folder=join(self.result_dire,self.dataset,self.features,self.arch,self.evaluation,self.config['MODEL_SIZE'])

        if not exists(pkl_folder):
            os.makedirs(pkl_folder)

        

        

        res={'y_true':y_true,
        'y_pred':y_pred,
        'y_true_cat':y_true_cat,
        'y_pred_cat':y_pred_cat,
        'history':history.history
        }

        with open(join(pkl_folder,'res.pickle'),'wb') as file:
            
            pickle.dump(res,file)

            
    
    def start_training(self):
        self.get_random_hp()
        self.do_training()
        