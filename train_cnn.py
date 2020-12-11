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
import read_settings
import datasets
import models
import matplotlib.pyplot as plt
import tensorflow as tf

#################################### Settings ####################################
## Read settings
global_settings = read_settings.check_global()
cnn_settings = read_settings.check_cnn()

# Input data
instrument = global_settings['input_data']['instrument']
data_dir = os.path.join('data', instrument)
split = global_settings['input_data']['split']
n_max = global_settings['input_data']['n_max']

# Random state
random_state = global_settings['random_state']

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


## Output
# Generate output directory pattern
if use_weights: # if using weigths 
    output_dir_patt = os.path.join('output', '_'.join(['cnn', 'w', instrument]))
else: # if not using weigths 
    output_dir_patt = os.path.join('output', '_'.join(['cnn', 'nw', instrument]))

# Look for previous outputs with same pattern
prev_output = glob.glob(output_dir_patt + '*')
# If an previous output exists, make a tar.gz archive
if prev_output:
    prev_output = prev_output[0]
    with tarfile.open(prev_output + '.tar.gz', "w:gz") as tar:
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
output_dir = '_'.join([output_dir_patt, datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")])
os.mkdir(output_dir)

# Write settings to output directory
read_settings.write_cnn_settings(global_settings, cnn_settings, output_dir)


##################################################################################

## Read data for CNN
df_train, df_valid, df_test, df_classes = datasets.read_data_cnn(
    path=os.path.join(data_dir, '_'.join([instrument, 'data.csv'])),
    split=split,
    n_max=n_max,
    random_state=random_state)

# Extract dataset composition (split by classif_id) and write it to output_dir
df_comp = pd.concat([
    pd.concat([df_train['classif_id'], pd.DataFrame({'split':['train'] * len(df_train)})], axis=1),
    pd.concat([df_valid['classif_id'], pd.DataFrame({'split':['valid'] * len(df_valid)})], axis=1),
    pd.concat([df_test['classif_id'], pd.DataFrame({'split':['test'] * len(df_test)})], axis=1)
], axis=0, ignore_index=True).groupby(['classif_id','split']).size().unstack(fill_value=0)
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
        class_weights.update({i : count_max / idx[1]})


## Generate batches
train_batches = datasets.DataGenerator(
    df=df_train,
    data_dir=data_dir,
    batch_size=batch_size, 
    augment=augment,
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

## Generate CNN
my_cnn = models.create_cnn(
    fc_layers_nb,
    fc_layers_dropout, 
    fc_layers_size, 
    classif_layer_dropout, 
    classif_layer_size=nb_classes, 
    train_fe=train_fe, 
    glimpse=True)

## Compile CNN
my_cnn = models.compile_cnn(
    my_cnn, 
    initial_lr, 
    steps_per_epoch=len(train_batches)//epochs, 
    lr_method=lr_method, 
    decay_rate=decay_rate, 
    loss=loss
)

## Train CNN
history = models.train_cnn(
    model=my_cnn, 
    train_batches=train_batches, 
    valid_batches=valid_batches, 
    batch_size=batch_size, 
    epochs=epochs, 
    class_weights=class_weights, 
    output_dir=output_dir
)

## Predict test batches and evaluate CNN
test_accuracy, test_loss = models.predict_evaluate_cnn(
    model=my_cnn, 
    batches=test_batches, 
    df_classes=df_classes, 
    output_dir=output_dir
)


