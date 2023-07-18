

import tensorflow as tf


def features_extraction_model(resized_height=350,resized_width=350):
    ## defining the DenseNet121 model pretrained with imagenet dataset
    feature_extractor=tf.keras.applications.DenseNet121(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(resized_height,resized_width,3),
    pooling='max',

)   

    for layer in feature_extractor.layers[:-15]:
        layer.trainable=False
    
    ##Defining the input size for single image                    
    inputs=tf.keras.Input((resized_height,resized_width,3))
    preprocessed=tf.keras.applications.densenet.preprocess_input(inputs)

    ## Give features for each image
    outputs=feature_extractor(preprocessed)
    ## defining feature extractor model
    model=tf.keras.Model(inputs,outputs,name='feature_extractor')

    
    return model

