import os
import glob
import lycon
import cv2
import pandas as pd
import numpy as np
import read_settings
import datasets
import model_cnn
import matplotlib.pyplot as plt
import tensorflow as tf

#################################### Settings ####################################
## Read settings
global_settings = read_settings.check_global()
cnn_settings = read_settings.check_cnn()

# Input data
instrument = global_settings['input_data']['instrument']
data_dir = os.path.join('data', instrument)

# Random state
random_state = global_settings['random_state']

# Output
output_dir = '_'.join(['output_cnn', instrument])
if global_settings['delete_previous']:
    files = glob.glob(os.path.join(output_dir, '*'))
    for f in files:
        os.remove(f)

# CNN settings
batch_size    = cnn_settings['data']['batch_size']
px_del        = cnn_settings['data']['px_del']
preserve_size = cnn_settings['data']['preserve_size']
augment       = cnn_settings['data']['augment']
use_weights   = cnn_settings['data']['use_weights']

fc_layers_nb          = cnn_settings['architecture']['fc_layers_nb']
fc_layers_size        = cnn_settings['architecture']['fc_layers_size']
fc_layers_dropout     = cnn_settings['architecture']['fc_layers_dropout']
classif_layer_dropout = cnn_settings['architecture']['classif_layer_dropout']
train_fe              = cnn_settings['architecture']['train_fe']

lr_method  = cnn_settings['compilation']['lr_method']
initial_lr = cnn_settings['compilation']['initial_lr']
decay_rate = cnn_settings['compilation']['decay_rate']
loss       = cnn_settings['compilation']['loss']

epochs = cnn_settings['training']['epochs']

##################################################################################

## Read data for CNN
df_train, df_valid, df_test = datasets.read_data_cnn(
    path = os.path.join(data_dir, '_'.join([instrument, 'data.csv'])),
    random_state=random_state)

# Write train, valid and test splits to output directory for future inspection
df_train.to_csv(os.path.join(output_dir, 'df_train.csv'), index=False)
df_valid.to_csv(os.path.join(output_dir, 'df_valid.csv'), index=False)
df_test.to_csv(os.path.join(output_dir, 'df_test.csv'), index=False)

# Number of plankton classes to predict
nb_classes = df_train['classif_id'].nunique()
classes = df_train['classif_id'].unique()


## Generate batches
train_batches = datasets.DataGenerator(
    df=df_train,
    data_dir=data_dir,
    batch_size=batch_size, 
    px_del=px_del)

valid_batches = datasets.DataGenerator(
    df=df_valid,
    data_dir=data_dir,
    batch_size=batch_size, 
    px_del=px_del)

test_batches = datasets.DataGenerator(
    df=df_test,
    data_dir=data_dir,
    batch_size=batch_size, 
    px_del=px_del)

for image_batch, label_batch in train_batches:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

# glimpse at batch
#if glimpse:
#    datasets.batch_glimpse(test_batches, classes)

    


## Generate CNN
my_cnn = model_cnn.create_cnn(
    fc_layers_nb,
    fc_layers_dropout, 
    fc_layers_size, 
    classif_layer_dropout, 
    classif_layer_size=nb_classes, 
    train_layers='all', 
    glimpse=False)

## Compile CNN
my_cnn = model_cnn.compile_cnn(
    my_cnn, 
    initial_lr, 
    steps_per_epoch=len(train_batches)//epochs, 
    lr_method=lr_method, 
    decay_rate=decay_rate, 
    loss=loss
)

## Train CNN
history = model_cnn.train_cnn(my_cnn, train_batches, valid_batches, batch_size, epochs, output_dir)
    
## Predict test batches and evaluate CNN
accuracy, loss = model_cnn.predict_evaluate_cnn(my_cnn, test_batches, classes, output_dir)
