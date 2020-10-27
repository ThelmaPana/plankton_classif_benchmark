import os
import lycon
import cv2
import pandas as pd
import numpy as np
import datasets
import model_cnn
#import matplotlib.pyplot as plt

#################################### Settings ####################################
## Data
instrument = "isiis"
data_dir = os.path.join('data', instrument)
random_state = 42
batch_size = 32
image_dimensions = (224, 224, 3)
shuffle=True
#augment=False #TODO
px_del = 0
preserve_size=False

## CNN model
# architecture
fc_layers_nb = 2
fc_layers_size = 1280
fc_layers_dropout = 0.4
classif_layer_size = 22
classif_layer_dropout = 0.2
train_layers = 'all'

# Training
initial_epochs = 50
#initial_epochs = 1
steps_per_epoch = 500//batch_size

# Compilation
lr_method = 'decay' # or could be 'constant'
initial_lr = 0.001
decay_rate = 0.97
loss = 'sfce' # or 'sfce'

##################################################################################

## Read data for CNN
df_train, df_valid, df_test = datasets.read_data_cnn(
    path = os.path.join(data_dir, 'isiis_data.csv'),
    random_state=random_state)


# Number of plankton classes to predict
nb_classes = df_train['classif_id'].nunique()
classes = df_train['classif_id'].unique()


## Generate batches
train_batches = datasets.DataGenerator(
    df=df_train,
    data_dir=data_dir,
    batch_size=batch_size, 
    px_del=px_del, 
    shuffle=shuffle)

valid_batches = datasets.DataGenerator(
    df=df_valid,
    data_dir=data_dir,
    batch_size=batch_size, 
    px_del=px_del, 
    shuffle=shuffle)

test_batches = datasets.DataGenerator(
    df=df_test,
    data_dir=data_dir,
    batch_size=batch_size, 
    px_del=px_del, 
    shuffle=shuffle)

for image_batch, label_batch in train_batches:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

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
    steps_per_epoch, 
    lr_method='constant', 
    decay_rate=None, 
    loss='cce'
)