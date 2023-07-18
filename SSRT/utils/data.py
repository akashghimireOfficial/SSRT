# Copyright 2021 Simone Angarano. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
import math


def random_flip(x):
    
    time_steps = x.shape[0]
    n_features = x.shape[1]
    x = tf.reshape(x, (time_steps, n_features//3, 3))
    
    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    if choice >= 0.5:
        x = tf.math.multiply(x, [-1.0,1.0,1.0])
        
    x = tf.reshape(x, (time_steps,n_features))
    return x







def rotate(x):
    anglesx=np.arange(15,360,15)
    anglex=(np.pi/180.0)*np.random.choice(anglesx)
    sinex=math.sin(anglex)
    cosx=math.cos(anglex)
    

    anglesy=np.arange(15,360,15)
    angley=(np.pi/180.0)*np.random.choice(anglesy)
    siney=math.sin(angley)
    cosy=math.cos(angley)

    anglesz=np.arange(15,360,15)
    anglez=(np.pi/180.0)*np.random.choice(anglesz)
    sinez=math.sin(anglez)
    cosz=math.cos(anglez)
    
    

    ##rotate matrix

    # rt_mt=tf.constant([[0.86,-0.5,0],
    # [0.5,0.86,0],
    # [0,0,1]],dtype=tf.float64)

    rtx=tf.constant([[1,0,0],
    [0,cosx,-sinex],
    [0,sinex,cosx]],dtype=tf.float64)

    rty=tf.constant([[cosy,0,siney],
    [0,1,0],
    [-siney,0,cosy]],dtype=tf.float64)
    
    
    rtz=tf.constant([[cosz,-sinez,0],
    [sinez,cosz,0],
    [0,0,1]],dtype=tf.float64)
    
    #rt_mt=tf.matmul(rtx,rty,rtz)

    time_steps = x.shape[0]
    n_features = x.shape[1]
    x = tf.reshape(x, (time_steps,  3, n_features//3))
    
    choicex = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    choicey = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    choicez = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    if choicex >0.8:

        x=tf.matmul(rtx,x)
    
    if choicey >0.75:

        x=tf.matmul(rty,x)

    if choicez >0.65:

        x=tf.matmul(rtz,x)
    
    
    
    x = tf.reshape(x, (time_steps, n_features))

    return x










def random_noise(x, y):
    time_steps = tf.shape(x)[0]
    n_features = tf.shape(x)[1]
    noise = tf.random.normal((time_steps, n_features), mean=0.0, stddev=0.05, dtype=tf.float64)
    x = x + noise
    return x, y



def aug(x,y):

    x=random_flip(x)
    x=rotate(x)
    #x,y=random_noise(x,y)
    #x=shear(x)
    return x,y






def one_hot(x, y):
    y=tf.cast(y,dtype=tf.int16)
    return x, tf.one_hot(y,31)

# def combine_inputs(X_pose, X_rgb, y):
#     return {"input_1": X_pose, "input_2": X_rgb}, y


def combine_inputs(input1, input2, label1,label2):
    return (input1, input2), (label1,label2)

def uncombine_inputs(*inputs_label):
    inputs, labels = inputs_label
    input1, input2 = inputs
    label1,label2=labels
    return input1, input2, label1,label2