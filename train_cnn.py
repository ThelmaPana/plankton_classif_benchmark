#!/usr/bin/env python

import os
import glob
import lycon
import cv2
import pandas as pd
import tarfile
import shutil
import datetime
import numpy as np
import math
import pickle

import read_settings
import datasets
import models
import matplotlib.pyplot as plt
import tensorflow as tf


#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.config.experimental.set_virtual_device_configuration(gpus[0],
#    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20480)])

#################################### Settings ####################################
## Read settings
global_settings = read_settings.check_global()
cnn_settings = read_settings.check_cnn()

# Input data
instrument = global_settings['input_data']['instrument']
data_dir = os.path.join('data', instrument)
frac = global_settings['input_data']['frac']

# Random state
random_state = global_settings['random_state']

# CNN settings
batch_size    = cnn_settings['data']['batch_size']
px_del        = cnn_settings['data']['px_del']
preserve_size = cnn_settings['data']['preserve_size']
augment       = cnn_settings['data']['augment']
use_weights   = cnn_settings['data']['use_weights']
weights       = cnn_settings['data']['weights']

fc_layers_nb          = cnn_settings['architecture']['fc_layers_nb']
fc_layers_size        = cnn_settings['architecture']['fc_layers_size']
fc_layers_dropout     = cnn_settings['architecture']['fc_layers_dropout']
classif_layer_dropout = cnn_settings['architecture']['classif_layer_dropout']
train_fe              = cnn_settings['architecture']['train_fe']

lr_method  = cnn_settings['compilation']['lr_method']
initial_lr = cnn_settings['compilation']['initial_lr']
decay_rate = cnn_settings['compilation']['decay_rate']
loss       = cnn_settings['compilation']['loss']

resume  = cnn_settings['training']['resume']
epochs  = cnn_settings['training']['epochs']
workers = cnn_settings['training']['workers']


## Output
# Generate output directory pattern
if use_weights: # if using weigths 
    output_dir_patt = os.path.join('output', '_'.join(['cnn', 'w', instrument]))
else: # if not using weigths 
    output_dir_patt = os.path.join('output', '_'.join(['cnn', 'nw', instrument]))

# Look for previous outputs with same pattern
prev_output = glob.glob(output_dir_patt + '*')
prev_output.sort()

# Case of not resuming from previous CNN training
if not resume:
    # If an previous output exists, make a tar.gz archive
    if prev_output:
        prev_output = prev_output[0]
        with tarfile.open(prev_output + '.tar.gz', 'w:gz') as tar:
            tar.add(prev_output, arcname=os.path.basename(prev_output))
            tar.close()
        
        # Delete directory with old output
        shutil.rmtree(prev_output)
         
        # Check if a directory exists for old outputs
        old_output_dir = os.path.join('output', 'old')
        # If it does not exist, create it
        if not os.path.exists(old_output_dir):
            os.makedirs(old_output_dir)
    
        # Move tar.gz file with old outputs
        shutil.move(prev_output + '.tar.gz', os.path.join(old_output_dir, os.path.basename(prev_output) + '.tar.gz'))
        
    # Create a new output directory
    output_dir = '_'.join([output_dir_patt, datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')])
    print(f'Creating new output directory {output_dir}')
    os.mkdir(output_dir)

# Case of resuming from previous CNN training
else:
    # Check that settings are similar as previous ones
    read_settings.check_previous_cnn_settings(global_settings, cnn_settings, prev_output)
    
    # Look for saved model in most recent output
    saved_models = glob.glob(os.path.join(prev_output[-1], 'model.last.epoch.*.hdf5'))
    saved_models.sort()
    
    # Case of an existing previous output to resume from
    if len(saved_models) > 0:
        # Set output dir to the most recent previous output
        output_dir = prev_output[-1]
        # Choose most recent model
        saved_model = saved_models[-1]
        print(f'Resuming training from {output_dir}, found saved model {os.path.basename(saved_model)}')
        
    # Case of no existing previous output to resume from
    else:
        # Create a new output directory
        output_dir = '_'.join([output_dir_patt, datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')])
        print(f'No previous training to resume from, creating new output directory {output_dir}')
        os.mkdir(output_dir)
        
# Write settings to output directory
read_settings.write_cnn_settings(global_settings, cnn_settings, output_dir)


##################################################################################

## Read data for CNN
df_train, df_valid, df_test, df_classes, df_comp = datasets.read_data_cnn(
    path=os.path.join(data_dir, '_'.join([instrument, 'data.csv'])),
    frac=frac,
    random_state=random_state
)

# Write dataset composition to output_dir
df_comp.to_csv(os.path.join(output_dir, 'df_comp.csv'), index=True)

# Number of plankton classes to predict
nb_classes = len(df_classes)

# Generate class weights
class_weights = None
if use_weights:
    class_counts = df_train.groupby('classif_id').size()
    count_max = 0
    class_weights = {}
    for idx in class_counts.items():
        count_max = (idx[1], count_max) [idx[1] < count_max]
    for i,idx in enumerate(class_counts.items()):
        if weights == 'i_f': # Weights computed with inverse frequency
            class_weights.update({i : count_max / idx[1]})
        elif weights == 'sqrt_i_f': # # Weights computed with square root of inverse frequency
            class_weights.update({i : math.sqrt(count_max / idx[1])})


## Generate batches
train_batches = datasets.DataGenerator(
    df=df_train,
    classes=df_classes.classif_id.tolist(),
    data_dir=data_dir,
    batch_size=batch_size, 
    augment=augment,
    px_del=px_del,
    random_state=random_state
)  

valid_batches = datasets.DataGenerator(
    df=df_valid,
    classes=df_classes.classif_id.tolist(),
    data_dir=data_dir,
    batch_size=batch_size, 
    augment=False, # do not augment or shuffle validation data
    shuffle=False,
    px_del=px_del,
    random_state=random_state
)

test_batches = datasets.DataGenerator(
    df=df_test,
    classes=df_classes.classif_id.tolist(),
    data_dir=data_dir,
    batch_size=batch_size, 
    augment=False, # do not augment or shuffle test data
    shuffle=False,
    px_del=px_del,
    random_state=random_state
)

for image_batch, label_batch in train_batches:
    print('Image batch shape: ', image_batch.shape)
    print('Label batch shape: ', label_batch.shape)
    break


## Case of not resuming from previous training
if not resume:
    ## Generate CNN
    my_cnn = models.create_cnn(
        fc_layers_nb,
        fc_layers_dropout, 
        fc_layers_size, 
        classif_layer_dropout, 
        classif_layer_size=nb_classes, 
        train_fe=train_fe, 
        glimpse=True
    )
    
    ## Compile CNN
    my_cnn = models.compile_cnn(
        my_cnn, 
        lr_method=lr_method, 
        initial_lr=initial_lr, 
        steps_per_epoch=len(train_batches), 
        decay_rate=decay_rate, 
        loss=loss
    )
    
    # Set firts epoch to 0
    initial_epoch = 0
    # Declare no previous training history
    prev_history = None


## Case of resuming from previous training
else:
    my_cnn, initial_epoch, prev_history = models.load_cnn(saved_model, glimpse = True)


## Train CNN
history, best_epoch = models.train_cnn(
    model=my_cnn, 
    prev_history=prev_history,
    train_batches=train_batches, 
    valid_batches=valid_batches, 
    batch_size=batch_size, 
    initial_epoch=initial_epoch,
    epochs=epochs+initial_epoch, 
    class_weights=class_weights, 
    output_dir=output_dir,
    workers=workers,
)

    
## Predict test batches and evaluate CNN
models.predict_evaluate_cnn(
    model=my_cnn, 
    best_epoch=best_epoch,
    batches=test_batches, 
    true_classes = np.array(df_test.classif_id.tolist()),
    df_classes=df_classes, 
    output_dir=output_dir,
    workers=workers,
)
