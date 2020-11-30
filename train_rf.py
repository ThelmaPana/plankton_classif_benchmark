#!/usr/bin/env python

import os
import glob
import read_settings
import datasets
import models
import pandas as pd
import tarfile
import shutil
import datetime
from plotnine import *
from sklearn.metrics import accuracy_score

#################################### Settings ####################################       
## Read settings
global_settings = read_settings.check_global()
rf_settings = read_settings.check_rf()

# Input data
instrument = global_settings['input_data']['instrument']
data_dir = os.path.join('data', instrument)
split = global_settings['input_data']['split']
n_max = global_settings['input_data']['n_max']

# Random state
random_state = global_settings['random_state']

# Output
output_dir_pat = os.path.join('output', '_'.join(['rf', instrument]))

# Look for previous output
prev_output = glob.glob(output_dir_pat + '*')
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
output_dir = os.path.join('output', '_'.join(['rf', instrument, datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")]))
os.mkdir(output_dir)

# Write settings to output directory
read_settings.write_rf_settings(global_settings, rf_settings, output_dir)

# RF settings
n_jobs = rf_settings['n_jobs'] 

max_features_try = rf_settings['grid_search']['max_features_try']
min_samples_leaf_try = rf_settings['grid_search']['min_samples_leaf_try']
n_estimators_try = rf_settings['grid_search']['n_estimators_try']

max_features = rf_settings['hyperparameters']['max_features']
min_samples_leaf = rf_settings['hyperparameters']['min_samples_leaf']
n_estimators = rf_settings['hyperparameters']['n_estimators']
        
##################################################################################

## Read data for RF
df_train, df_valid, df_test, df_classes = datasets.read_data_rf(
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


## Grid search
# Do grid serach
if rf_settings['grid_search']['go']:
    gs_results, best_params = models.gridsearch_rf(
        df_train, 
        df_valid, 
        max_features_try=max_features_try,
        min_samples_leaf_try=min_samples_leaf_try,
        n_estimators_try=n_estimators_try,
        output_dir=output_dir,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    # Set parameters for future RF models
    n_estimators = best_params['n_estimators']
    max_features = best_params['max_features']
    min_samples_leaf = best_params['min_samples_leaf']
    print(f'Selected parameters are: n_estimators = {n_estimators}, max_features = {max_features}, min_samples_leaf = {min_samples_leaf}')


## Fit the RF of training data
rf = models.train_rf(
    df=df_train, 
    n_estimators=n_estimators, 
    max_features=max_features, 
    min_samples_leaf=min_samples_leaf, 
    n_jobs=n_jobs, 
    random_state=random_state
)


## Evaluate the RF on test data
test_accuracy = models.predict_evaluate_rf(
    rf_model=rf, 
    df=df_test,
    df_classes=df_classes,
    output_dir = output_dir
)


