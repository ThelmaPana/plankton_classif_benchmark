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

epochs = cnn_settings['training']['epochs']
workers = cnn_settings['training']['workers']


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
df_train_cnn, df_valid_cnn, df_test_cnn, df_classes = datasets.read_data_cnn(
    path=os.path.join(data_dir, '_'.join([instrument, 'data.csv'])),
    split=split,
    n_max=n_max,
    random_state=random_state)


## Read data for RF
df_train_rf, df_valid_rf, df_test_rf, df_classes = datasets.read_data_rf(
    path=os.path.join(data_dir, '_'.join([instrument, 'data.csv'])),
    split=split,
    n_max=n_max,
    random_state=random_state)

df_all_rf = pd.concat([
    pd.concat([df_train_rf, pd.DataFrame({'split':['train'] * len(df_train_rf)})], axis=1),
    pd.concat([df_valid_rf, pd.DataFrame({'split':['valid'] * len(df_valid_rf)})], axis=1),
    pd.concat([df_test_rf, pd.DataFrame({'split':['test'] * len(df_test_rf)})], axis=1)
], axis=0, ignore_index=True)

df_all_rf = df_all_rf[['area', 'filled_area', 'equivalent_diameter', 'perimeter', 'split', 'classif_id']]
df_all_rf.describe()
