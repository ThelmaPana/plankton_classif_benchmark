#!/usr/bin/env python

import os
import glob
import read_settings
import datasets
import models
import pandas as pd
import math
import tarfile
import shutil
import datetime
#from plotnine import *

#################################### Settings ####################################       
## Read settings
global_settings = read_settings.check_global()
rf_settings = read_settings.check_rf()

# Input data
instrument = global_settings['input_data']['instrument']
data_dir = os.path.join('data', instrument)
frac = global_settings['input_data']['frac']

# Random state
random_state = global_settings['random_state']

# RF settings
n_jobs      = rf_settings['n_jobs'] 
use_weights = rf_settings['use_weights'] 
weights     = rf_settings['weights']

eval_metric          = rf_settings['grid_search']['eval_metric']
max_features_try     = rf_settings['grid_search']['max_features_try']
min_samples_leaf_try = rf_settings['grid_search']['min_samples_leaf_try']
n_estimators_try     = rf_settings['grid_search']['n_estimators_try']

max_features     = rf_settings['hyperparameters']['max_features']
min_samples_leaf = rf_settings['hyperparameters']['min_samples_leaf']
n_estimators     = rf_settings['hyperparameters']['n_estimators']


## Output
# Generate output directory pattern
if use_weights: # if using weigths 
    output_dir_patt = os.path.join('output', '_'.join(['rf', 'w', instrument]))
else: # if not using weigths 
    output_dir_patt = os.path.join('output', '_'.join(['rf', 'nw', instrument]))

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
read_settings.write_rf_settings(global_settings, rf_settings, output_dir)


##################################################################################

## Read data for RF
df_train, df_valid, df_test, df_classes, df_comp = datasets.read_data_rf(
    path=os.path.join(data_dir, '_'.join([instrument, 'data.csv'])),
    frac=frac,
    random_state=random_state
)

# Write dataset composition to output_dir
df_comp.to_csv(os.path.join(output_dir, 'df_comp.csv'), index=True)

# Generate class weights
class_weights = None
if use_weights:
    class_counts = df_train.groupby('classif_id').size()
    count_max = 0
    class_weights = {}
    for idx in class_counts.items():
        count_max = (idx[1], count_max) [idx[1] < count_max]
    for i,idx in enumerate(class_counts.items()):
        if weights == 'i_f':
            class_weights.update({idx[0] : count_max / idx[1]})
        elif weights == 'sqrt_i_f':
            class_weights.update({idx[0] : math.sqrt(count_max / idx[1])})
    
## Grid search
# Do grid serach
if rf_settings['grid_search']['go']:
    gs_results, best_params = models.gridsearch_rf(
        df_train, 
        df_valid, 
        classes = df_classes.classif_id.tolist(),
        eval_metric=eval_metric,
        max_features_try=max_features_try,
        min_samples_leaf_try=min_samples_leaf_try,
        n_estimators_try=n_estimators_try,
        output_dir=output_dir,
        n_jobs=n_jobs,
        class_weights=class_weights,
        random_state=random_state
    )
    
    # Set parameters for future RF models
    n_estimators = best_params['n_estimators']
    max_features = best_params['max_features']
    min_samples_leaf = best_params['min_samples_leaf']


## Fit the RF of training data
rf = models.train_rf(
    df=df_train, 
    n_estimators=n_estimators, 
    max_features=max_features, 
    min_samples_leaf=min_samples_leaf, 
    n_jobs=n_jobs, 
    class_weights=class_weights,
    random_state=random_state
)


## Evaluate the RF on test data
models.predict_evaluate_rf(
    rf_model=rf, 
    df=df_test,
    df_classes=df_classes,
    output_dir=output_dir
)
