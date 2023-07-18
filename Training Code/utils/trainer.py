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
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# OPTUNA
import optuna
from optuna.trial import TrialState
# CUSTOM LIBRARIES

from utils.generator import FrameGenerator,PoseGenerator,Fusiongenerator,PoseGenerator_CD,Fusiongenerator_CD,FrameGenerator_CD

from utils.resnet183d import resnet3d
from utils.densenet import features_extraction_model
from utils.transformer import TransformerEncoder, pos_encoding,MultiHeadAttention,TransformerDecoder,Transformerfusion
#from utils.transformer_fus import Transformerfusion,TransformerEncoder

from utils.data import  random_flip, random_noise, one_hot,aug,combine_inputs,uncombine_inputs

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

            self.shape=(42,34)

        elif self.features=='rgb':
            self.backbone=self.config[self.features]['backbone']
            self.dataset_dire_rgb=join('features/',self.dataset,self.features,self.backbone,self.evaluation)
            self.shape=(42,2048)

        else:
            self.pose=config[self.features]['pose']
            
            
            
            self.dataset_dire_pose=join('features/',self.dataset,'skeleton',self.pose,self.evaluation)

            self.shape_pose=(42,34)

            self.backbone=self.config[self.features]['backbone']
            self.dataset_dire_rgb=join('features/',self.dataset,'rgb',self.backbone,self.evaluation,'crop')
            self.shape_rgb=(42,2048)

            self.arch=config[self.features]['arch']

            

            

            
        self.result_dire=self.config['Result_dire']    




        self.bin_path = self.config['MODEL_DIR']
        
        self.model_size = self.config['MODEL_SIZE']
        self.n_heads = self.config[self.model_size]['N_HEADS']
        #self.n_heads = 256
        self.n_layers = self.config[self.model_size]['N_LAYERS']
        
        self.dropout = self.config[self.model_size]['DROPOUT']
        self.mlp_head_size = self.config[self.model_size]['MLP']
        self.activation = tf.nn.gelu
        self.d_model = self.n_heads*1
        
        self.d_ff = self.d_model*2



    def build_act(self,transformer):
        inputs = tf.keras.layers.Input(shape=self.shape)
        

        

        

        x=tf.keras.layers.LSTM(units=512,return_sequences=True)(inputs)
        
        x=tf.keras.layers.LayerNormalization()(x)

        x=tf.keras.layers.LSTM(units=512,return_sequences=True)(x)
        
        x=tf.keras.layers.LayerNormalization()(x)

        x=tf.keras.layers.LSTM(128,return_sequences=False)(x)


        
        
        
        

    
        
        # x = transformer(inputs,mask=None,training=True)



        
        
        # x=tf.keras.layers.Flatten()(x)
        x=tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(self.mlp_head_size)(x)
        outputs = tf.keras.layers.Dense(self.num_class)(x)
        return tf.keras.models.Model(inputs, outputs)
        

    def build_fusion(self,transformer_fusion):
        input_pose = tf.keras.layers.Input(shape=self.shape_pose)
        input_rgb=  tf.keras.layers.Input(shape=self.shape_rgb)
        

        

        

        # x=tf.keras.layers.LSTM(units=self.d_model,return_sequences=True)(inputs)
        
        # x=tf.keras.layers.LayerNormalization()(x)

        
        

    
        
        

        
        pose,rgb=transformer_fusion([input_rgb,input_pose],training=True,mask=True)
        
        
       
        
        # pose_output = tf.keras.layers.Lambda(lambda x: x[:,0,:])(pose_output)
        # rgb_output = tf.keras.layers.Lambda(lambda x: x[:,0,:])(rgb_output)

       

        pose=tf.keras.layers.Flatten()(pose)
        pose=tf.keras.layers.Dropout(0.25)(pose)
        pose_output = tf.keras.layers.Dense(self.mlp_head_size)(pose)

        rgb=tf.keras.layers.Flatten()(rgb)
        rgb=tf.keras.layers.Dropout(0.25)(rgb)

        rgb_output = tf.keras.layers.Dense(self.mlp_head_size)(rgb)

        
        #x = tf.keras.layers.Dense(self.mlp_head_size)(x)
        pose_output = tf.keras.layers.Dense(self.num_class,name='pose')(pose_output)
        rgb_output = tf.keras.layers.Dense(self.num_class,name='rgb')(rgb_output)
        return tf.keras.models.Model([input_pose,input_rgb], [pose_output,rgb_output])


    def build_rgb_BERT(self,transformer):
        

        inputs=tf.keras.Input(shape=(42,350,350,3))

        x=tf.keras.layers.LSTM(units=512,return_sequences=True)(inputs)
        
        x=tf.keras.layers.LayerNormalization()(x)

        x=tf.keras.layers.LSTM(units=512,return_sequences=True)(x)
        
        x=tf.keras.layers.LayerNormalization()(x)

        x=tf.keras.layers.LSTM(128,return_sequences=False)(x)

        

        # x = transformer(x,mask=None,training=True)


        
        # x=tf.keras.layers.Flatten()(x)
        x=tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.mlp_head_size)(x)
        outputs = tf.keras.layers.Dense(self.num_class)(x)
        return tf.keras.models.Model(inputs, outputs)



    


    def get_model(self,trial_num):

       
        

        if self.features=='skeleton':
            transformer = TransformerEncoder(d_model=self.d_model,dff=self.d_ff,num_heads=self.n_heads,rate=self.dropout,num_layers=self.n_layers,pos=True,lstm_enc=False,pos_enc=True)
            self.model = self.build_act(transformer)

            # for layer in self.model.layers:
            #     if hasattr(layer, 'kernel_regularizer'):
            #         layer.kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.12, l2=0.12)

            
            lr = CustomSchedule(self.d_model, 
                    warmup_steps=2*self.config['N_EPOCHS']*self.config['WARMUP_PERC'],
                    decay_step=2*self.config['N_EPOCHS']*self.config['STEP_PERC'])

            optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=self.config['WEIGHT_DECAY'])

            self.model.compile(optimizer=optimizer,
                           loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)],
                           metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")
                           ]
                           
                           )
        elif self.features=='rgb':
            transformer = TransformerEncoder(d_model=self.d_model,dff=self.d_ff,num_heads=self.n_heads,rate=self.dropout,num_layers=self.n_layers,pos=True,lstm_enc=True,pos_enc=False)
            if self.backbone=='raw':
                
                
                self.model=self.build_rgb_BERT(transformer)
            else:
                self.model = self.build_act(transformer)

            lr = CustomSchedule(self.d_model, 
                warmup_steps=2*self.config['N_EPOCHS']*self.config['WARMUP_PERC'],
                decay_step=2*self.config['N_EPOCHS']*self.config['STEP_PERC'])

            optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=self.config['WEIGHT_DECAY'])

            self.model.compile(optimizer=optimizer,
                           loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)],
                           metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")
                           ]
                           
                           )


        else:
            
            transformer_fusion=Transformerfusion(d_model=self.d_model,dff=self.d_ff,num_heads=self.n_heads,rate=self.dropout,num_layers=self.n_layers)
            self.model=self.build_fusion(transformer_fusion)


            

            # for layer in self.model.layers:
            #     if hasattr(layer, 'kernel_regularizer'):
            #         layer.kernel_regularizer = tf.keras.regularizers.l1_l2(l1=0.12, l2=0.12)

        
            lr = CustomSchedule(self.d_model, 
                    warmup_steps=2*(self.config['N_EPOCHS'])*self.config['WARMUP_PERC'],
                    decay_step=2*(self.config['N_EPOCHS'])*self.config['STEP_PERC'])

            optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=self.config['WEIGHT_DECAY'])

            self.model.compile(optimizer=optimizer,
                                loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)],
                                
                                metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
                                
                                )

        
        



        # lr = CustomSchedule(self.d_model, 
        #      warmup_steps=1190*self.config['N_EPOCHS']*self.config['WARMUP_PERC'],
        #      decay_step=1190*self.config['N_EPOCHS']*self.config['STEP_PERC'])



                

        self.name_model_bin = self.config['MODEL_NAME'] + '_' + self.config['MODEL_SIZE'] + '_' + str(self.split) + '_' + str(self.fold)
        
        if self.features=='skeleton':

            self.new_bin_path=join(self.bin_path,self.dataset,self.features,self.pose,self.evaluation,self.config['MODEL_SIZE'],'{:04d}/'.format(trial_num))
        elif self.features=='rgb':
            self.new_bin_path=join(self.bin_path,self.dataset,self.features,self.backbone,self.evaluation,self.config['MODEL_SIZE'],'{:04d}/'.format(trial_num))

        else:

            self.new_bin_path=join(self.bin_path,self.dataset,self.features,self.arch,self.evaluation,self.config['MODEL_SIZE'],'{:04d}/'.format(trial_num))


        

        if self.features=='fusion':


            self.checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=self.new_bin_path+self.name_model_bin,
                                                               monitor="val_rgb_accuracy",
                                                               save_best_only=True,
                                                               mode='max',
                                                               save_weights_only=True,
                                                        )

            
        else:
            self.checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=self.new_bin_path+self.name_model_bin,
                                                               monitor="val_accuracy",
                                                               save_best_only=True,
                                                               mode='max',
                                                               save_weights_only=True
                                                        )

           


       

    def get_data_pose(self):

        
        
        batch_size=self.config['BATCH_SIZE']
        train_dire=join(self.dataset_dire_pose,'train')
        val_dire=join(self.dataset_dire_pose,'val')
        test_dire=join(self.dataset_dire_pose,'test')

    	
        
        output_signature = (tf.TensorSpec(shape = (None, 34), dtype = tf.float32),
                    tf.TensorSpec(shape = self.num_class, dtype = tf.int16))
        
        


        self.ds_train_pose=tf.data.Dataset.from_generator(PoseGenerator(pose_path=train_dire,training=True),output_signature=output_signature)
        self.ds_train_pose=self.ds_train_pose.cache('cache/{}/cskeleton/train/'.format(self.evaluation))
        self.ds_train_pose=self.ds_train_pose.batch(batch_size)
        self.ds_train_pose=self.ds_train_pose.prefetch(tf.data.experimental.AUTOTUNE)

        self.ds_test_pose=tf.data.Dataset.from_generator(PoseGenerator_CD(pose_path=test_dire,training=False),output_signature=output_signature)
        self.ds_test_pose=self.ds_test_pose.cache('cache/{}/cskeleton/test/'.format(self.evaluation))
        self.ds_test_pose=self.ds_test_pose.batch(batch_size)
        self.ds_test_pose=self.ds_test_pose.prefetch(tf.data.experimental.AUTOTUNE)

        self.ds_val_pose=tf.data.Dataset.from_generator(PoseGenerator(pose_path=val_dire,training=True),output_signature=output_signature)
        self.ds_val_pose=self.ds_val_pose.cache('cache/{}/cskeleton/val/'.format(self.evaluation))
        self.ds_val_pose=self.ds_val_pose.batch(batch_size)
        self.ds_val_pose=self.ds_val_pose.prefetch(tf.data.experimental.AUTOTUNE)



    def get_data_rgb(self):

        batch_size=self.config['BATCH_SIZE']
        train_dire=join(self.dataset_dire_rgb,'train')
        val_dire=join(self.dataset_dire_rgb,'val')
        test_dire=join(self.dataset_dire_rgb,'test')

        output_signature = (tf.TensorSpec(shape = (None,2048), dtype = tf.float32),
                tf.TensorSpec(shape = self.num_class, dtype = tf.int16))
    

        self.ds_train_rgb=tf.data.Dataset.from_generator(FrameGenerator(rgb_path=train_dire,training=True),output_signature=output_signature)
        self.ds_train_rgb=self.ds_train_rgb.cache('cache/{}/crgb/train/'.format(self.evaluation))
        self.ds_train_rgb=self.ds_train_rgb.batch(batch_size)
        self.ds_train_rgb=self.ds_train_rgb.prefetch(tf.data.experimental.AUTOTUNE)

        self.ds_test_rgb=tf.data.Dataset.from_generator(FrameGenerator_CD(rgb_path=test_dire,training=True),output_signature=output_signature)
        self.ds_test_rgb=self.ds_test_rgb.cache('cache/{}/crgb/test/'.format(self.evaluation))
        self.ds_test_rgb=self.ds_test_rgb.batch(batch_size)
        self.ds_test_rgb=self.ds_test_rgb.prefetch(tf.data.experimental.AUTOTUNE)

        self.ds_val_rgb=tf.data.Dataset.from_generator(FrameGenerator(rgb_path=val_dire,training=True),output_signature=output_signature)
        self.ds_val_rgb=self.ds_val_rgb.cache('cache/{}/crgb/val/'.format(self.evaluation))
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
                        tf.TensorSpec(shape = (None,2048), dtype = tf.float32),
                        tf.TensorSpec(shape = self.num_class, dtype = tf.int16),
                        tf.TensorSpec(shape = self.num_class, dtype = tf.int16))

        
        self.ds_train_fusion=tf.data.Dataset.from_generator(Fusiongenerator(pose_path=train_dire_pose,rgb_path=train_dire_rgb,training=True),output_signature=output_signature)
        self.ds_train_fusion=self.ds_train_fusion.map(combine_inputs)
        self.ds_train_fusion=self.ds_train_fusion.cache('cache/{}/cfusion/train'.format(self.evaluation))
        self.ds_train_fusion=self.ds_train_fusion.batch(batch_size)
        self.ds_train_fusion=self.ds_train_fusion.prefetch(tf.data.experimental.AUTOTUNE)


        self.ds_test_fusion=tf.data.Dataset.from_generator(Fusiongenerator_CD(pose_path=test_dire_pose,rgb_path=test_dire_rgb,training=False),output_signature=output_signature)
        self.ds_test_fusion=self.ds_test_fusion.map(combine_inputs)
        self.ds_test_fusion=self.ds_test_fusion.cache('cache/{}/cfusion/test/'.format(self.evaluation))
        self.ds_test_fusion=self.ds_test_fusion.batch(batch_size)
        self.ds_test_fusion=self.ds_test_fusion.prefetch(tf.data.experimental.AUTOTUNE)


        self.ds_val_fusion=tf.data.Dataset.from_generator(Fusiongenerator(pose_path=val_dire_pose,rgb_path=val_dire_rgb,training=True),output_signature=output_signature)
        self.ds_val_fusion=self.ds_val_fusion.map(combine_inputs)
        self.ds_val_fusion=self.ds_val_fusion.cache('cache/{}/cfusion/val/'.format(self.evaluation))
        self.ds_val_fusion=self.ds_val_fusion.batch(batch_size)
        self.ds_val_fusion=self.ds_val_fusion.prefetch(tf.data.experimental.AUTOTUNE)



        
    def get_random_hp(self):
        
        
        self.config['RN_STD'] = self.trial.suggest_discrete_uniform("RN_STD", 0.0, 0.05, 0.01)
        self.config['WEIGHT_DECAY'] = self.trial.suggest_discrete_uniform("WD", 1e-5, 1e-3, 1e-3)    
        self.config['N_EPOCHS'] = int(self.trial.suggest_discrete_uniform("EPOCHS",200,280,10))
        
        self.config['WARMUP_PERC'] = self.trial.suggest_discrete_uniform("WARMUP_PERC", 0.2, 0.4, 0.1)
        
        self.logger.save_log('\nRN_STD: {:.2e}'.format(self.config['RN_STD']))
        self.logger.save_log('EPOCHS: {}'.format(self.config['N_EPOCHS']))
        self.logger.save_log('WARMUP_PERC: {:.2e}'.format(self.config['WARMUP_PERC']))
        self.logger.save_log('WEIGHT_DECAY: {:.2e}\n'.format(self.config['WEIGHT_DECAY']))
        
    def do_training(self,trial_num):


        if self.features=='skeleton':
            self.get_data_pose()
            self.get_model(trial_num)
            tf.keras.backend.clear_session()
            history = self.model.fit(self.ds_train_pose,
                epochs=self.config['N_EPOCHS'], initial_epoch=0,
                validation_data=self.ds_val_pose,
                callbacks=[self.checkpointer], verbose=self.config['VERBOSE'])

              
            


            
            tf.keras.backend.clear_session()
            self.model.load_weights(self.new_bin_path+self.name_model_bin)
            
            X, y_true_cat = tuple(zip(*self.ds_test_pose))
            y_true_cat=tf.concat(y_true_cat, axis=0)
            y_pred_cat = tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1)
            y_true=np.argmax(y_true_cat,axis=1)
            y_pred=np.argmax(y_pred_cat,axis=1)

            accuracy_test=accuracy_score(y_true,y_pred)
       
            
            
            
            
            text = f"{self.config['MODEL_SIZE']}:  Accuracy Test: {accuracy_test} \n"

            
            print("Accuracy Result :",accuracy_test)
            
            
            self.logger.save_log(text)

            return accuracy_test,history.history

            #print('len :',len(self.ds_train))
        elif self.features=='rgb':
            
            self.get_data_rgb()

            

            self.get_model(trial_num)

            history = self.model.fit(self.ds_train_rgb,
                epochs=self.config['N_EPOCHS'], initial_epoch=0,
                validation_data=self.ds_val_rgb,
                callbacks=[self.checkpointer], verbose=self.config['VERBOSE'])

            tf.keras.backend.clear_session()
            self.model.load_weights(self.new_bin_path+self.name_model_bin)
            
            X, y_true_cat = tuple(zip(*self.ds_test_rgb))
            y_true_cat=tf.concat(y_true_cat, axis=0)
            y_pred_cat = tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1)
            y_true=np.argmax(y_true_cat,axis=1)
            y_pred=np.argmax(y_pred_cat,axis=1)

            accuracy_test=accuracy_score(y_true,y_pred)
       
            
            
            
            
            text = f"{self.config['MODEL_SIZE']}:  Accuracy Test: {accuracy_test} \n"

            
            print("Accuracy Result :",accuracy_test)
            
            
            self.logger.save_log(text)

            return accuracy_test,history.history


            

        else:
            self.get_data_fusion()

            self.get_model(trial_num)

            
            tf.keras.backend.clear_session()
            #self.model.load_weights(self.new_bin_path+self.name_model_bin)
            history = self.model.fit(self.ds_train_fusion,
                epochs=self.config['N_EPOCHS'], initial_epoch=0,
                validation_data=self.ds_val_fusion,
                callbacks=[self.checkpointer], verbose=self.config['VERBOSE'])

            
            
            # _, eval_accuracy_test_pose,eval_accuracy_test_rgb = self.model.evaluate(self.ds_test_fusion)    
            tf.keras.backend.clear_session()
            self.model.load_weights(self.new_bin_path+self.name_model_bin)

            
          
            
            y_pred_pose_cat,y_pred_rgb_cat = tf.nn.softmax(self.model.predict(self.ds_test_fusion))
            
            y_pred_cat=(y_pred_rgb_cat+y_pred_pose_cat) ## Summing the final fusion score

            y_pred=tf.argmax(y_pred_cat,axis=1)
            
            ds_test_labels = self.ds_test_fusion.map(lambda x, y: y[0])
            ds_test_labels=ds_test_labels.unbatch()
            

            
            y_true_cat = np.array(list(ds_test_labels.as_numpy_iterator()))

            y_true=tf.argmax(y_true_cat,axis=1)

            print('prediction_classes.shape :',y_pred.shape)
            print('ground_truth_labels.shape :',y_true.shape)

           

            accuracy_test =accuracy_score(y_true,y_pred)


            

            #print('Eval Accuracy :', eval_accuracy_test)

           





            
         

            text = f"{self.config['MODEL_SIZE']}:  Accuracy Test: {accuracy_test} \n"

            
            self.logger.save_log(text)

            return accuracy_test,history.history

 
            

        
        
        

        

      

        self.dense = tf.keras.layers.Dense(d_model)


    
         
    
    def start_training(self):
        self.get_random_hp()
        self.do_training()




    def objective(self, trial):
        self.trial = trial     
        self.get_random_hp()
        acc,_ = self.do_training(trial.number)
        return acc


    def evaluate(self, trial_num):

        if self.features=='skeleton':

            self.best_bin_path=join(self.bin_path,self.dataset,self.features,self.pose,self.evaluation,self.config['MODEL_SIZE'],'{:04d}/'.format(trial_num))
            tf.keras.backend.clear_session()
            self.model.load_weights(self.new_bin_path+self.name_model_bin)
            
            X, y_true_cat = tuple(zip(*self.ds_test_pose))
            y_true_cat=tf.concat(y_true_cat, axis=0)
            y_pred_cat = tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1)
            y_true=np.argmax(y_true_cat,axis=1)
            y_pred=np.argmax(y_pred_cat,axis=1)

            return y_true,y_pred,y_pred_cat
        elif self.features=='rgb':
            self.best_bin_path=join(self.bin_path,self.dataset,self.features,self.backbone,self.evaluation,self.config['MODEL_SIZE'],'{:04d}/'.format(trial_num))
            tf.keras.backend.clear_session()
            self.model.load_weights(self.best_bin_path+self.name_model_bin)
            
            X, y_true_cat = tuple(zip(*self.ds_test_rgb))
            y_true_cat=tf.concat(y_true_cat, axis=0)
            y_pred_cat = tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1)
            y_true=np.argmax(y_true_cat,axis=1)
            y_pred=np.argmax(y_pred_cat,axis=1)

            return y_true,y_pred,y_pred_cat

        else:

            self.best_bin_path=join(self.bin_path,self.dataset,self.features,self.arch,self.evaluation,self.config['MODEL_SIZE'],'{:04d}/'.format(trial_num))
            tf.keras.backend.clear_session()
            self.model.load_weights(self.best_bin_path+self.name_model_bin)

            
          
            
            y_pred_pose_cat,y_pred_rgb_cat = tf.nn.softmax(self.model.predict(self.ds_test_fusion))
            
            y_pred_cat=(y_pred_rgb_cat+y_pred_pose_cat) ## Summing the final fusion score

            y_pred=tf.argmax(y_pred_cat,axis=1)
            
            ds_test_labels = self.ds_test_fusion.map(lambda x, y: y[0])
            ds_test_labels=ds_test_labels.unbatch()
            

            
            y_true_cat = np.array(list(ds_test_labels.as_numpy_iterator()))

            y_true=tf.argmax(y_true_cat,axis=1)

            return y_true,y_pred,y_pred_cat


        
 
       
       
        

    
        
    def do_random_search(self):
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(study_name='{}_random_search'.format(self.config['MODEL_NAME']),
                                         direction="maximize", pruner=pruner)
        self.study.optimize(lambda trial: self.objective(trial),
                            n_trials=self.config['N_TRIALS'])

        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        self.logger.save_log("Study statistics: ")
        self.logger.save_log(f"  Number of finished trials: {len(self.study.trials)}")
        self.logger.save_log(f"  Number of pruned trials: {len(pruned_trials)}")
        self.logger.save_log(f"  Number of complete trials: {len(complete_trials)}")

        self.logger.save_log("Best trial:")

        self.logger.save_log(f"  Value: {self.study.best_trial.value}")

        self.logger.save_log("  Params: ")
        for key, value in self.study.best_trial.params.items():
            self.logger.save_log(f"    {key}: {value}")

        joblib.dump(self.study,
          f"{self.config['RESULTS_DIR']}/{self.config['MODEL_NAME']}_random_search_{str(self.study.best_trial.value)}.pkl")


        if self.features=='skeleton':

            pkl_folder=join(self.result_dire,self.dataset,self.features,self.pose,self.evaluation,self.config['MODEL_SIZE'])
            txt_dire=join(self.result_dire,self.dataset,self.features,self.pose,self.evaluation,self.config['MODEL_SIZE'],'best.txt')
        elif self.features=='rgb':
            pkl_folder=join(self.result_dire,self.dataset,self.features,self.backbone,self.evaluation,self.config['MODEL_SIZE'])
            txt_dire=join(self.result_dire,self.dataset,self.features,self.backbone,self.evaluation,self.config['MODEL_SIZE'],'best.txt')

        else:

            pkl_folder=join(self.result_dire,self.dataset,self.features,self.arch,self.evaluation,self.config['MODEL_SIZE'])
            txt_dire=join(self.result_dire,self.dataset,self.features,self.arch,self.evaluation,self.config['MODEL_SIZE'],'best.txt')

        
        if not exists(pkl_folder):
            os.makedirs(pkl_folder)

        with open(txt_dire,'w') as f:
            f.write('{:04d}'.format(self.study.best_trial.number))

        y_true,y_pred,y_pred_cat=self.evaluate(self.study.best_trial.number)

        res={'y_true':y_true,
        'y_pred':y_pred,
        'y_pred_cat':y_pred_cat
        }

        with open(join(pkl_folder,'res.pickle'),'wb') as file:
            pickle.dump(res,file)
        