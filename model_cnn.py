import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
#from tensorflow.keras.losses import Reduction
from tensorflow.keras import layers#, applications, models, backend, experimental



def create_cnn(fc_layers_nb, fc_layers_dropout, fc_layers_size, classif_layer_dropout, classif_layer_size,  train_layers = 'head', glimpse = False):

    """
    Generates a CNN model. 
    
    Args:
        fc_layers_nb (int): number of fully connected layers 
        fc_layers_size (int): size of fully connected layers 
        fc_layers_dropout (float): dropout of fully connected layers 
        classif_layer_size (int): size of classification layer (i.e. number of classes to predict)
        classif_layer_dropout (float): dropout of classification layer
        train_layers (str): whether to train only classification head ('head') or all layers ('all')
        glimpse(bool): whether to show a model summary
    
    Returns:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNN model
        
    """
    
    ## Initiate empty model
    model = tf.keras.Sequential()
    
    ## MobileNet V2 feature extractor
    fe_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    fe_layer = hub.KerasLayer(fe_url, input_shape=(224, 224, 3))
    # If all layers should be trained, set feature extractor to trainable
    if train_layers == 'all':
        fe_layer.trainable=True
    model.add(fe_layer)
    
    ## Fully connected layers
    for i in range(fc_layers_nb):
        model.add(layers.Dropout(fc_layers_dropout))
        model.add(layers.Dense(fc_layers_size, activation='relu'))
    
    ### Classification layers
    model.add(layers.Dropout(classif_layer_dropout))
    model.add(layers.Dense(classif_layer_size))
    
    if glimpse:
        model.summary()

    return model

