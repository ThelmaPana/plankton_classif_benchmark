import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def read_data_cnn(path, random_state=None):
    """
    Read a csv file containing data to train the cnn
    
    Args:
        path (str): path to the file
    
    Returns:
        df_train (DataFrame): training data containing path to image and classif_id
        df_val (DataFrame): validation data containing path to image and classif_id
        df_test (DataFrame): testing data containing path to image and classif_id
    """
    
    df = pd.read_csv(path)
    
    # TODO check that mandatory columns are present: 'object_id', 'classif_id', 'path_to_img'
    
    # The classifier is a CNN, keep 'classif_id', and 'path_to_img'
    df = df[['path_to_img', 'classif_id']]
    
    # Make a stratified sampling by classif_id: 70% of data for training, 15% for validation and 15% for test
    y = df.pop('classif_id')
    X = df
    # 70% of data for training
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
    # split remaining 30% to 50-50 for validation and testing
    X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=0.5, random_state=random_state, stratify=y_eval)
    
    # put back together X and y for training, validation and test dataframes
    df_train = X_train.copy()
    df_train['classif_id'] = y_train
    df_train = df_train.sort_values('classif_id', axis=0).reset_index(drop=True)

    df_val = X_val.copy()
    df_val['classif_id'] = y_val
    df_val = df_val.sort_values('classif_id', axis=0).reset_index(drop=True)
    
    df_test = X_test.copy()
    df_test['classif_id'] = y_test
    df_test = df_test.sort_values('classif_id', axis=0).reset_index(drop=True)
  
    return df_train, df_val, df_test


def read_data_rf(path, random_state=None):
    """
    Read a csv file containing data to train the RandomForest and scale features between 0 and 1. 
    
    Args:
        path (str): path to the file
    
    Returns:
        df_train (DataFrame): training data containing object features and classif_id
        df_val (DataFrame): validation data containing object features and classif_id
        df_test (DataFrame): testing data containing object features and classif_id
    """
    
    df = pd.read_csv(path)
    
    # TODO check that mandatory columns are present: 'object_id', 'classif_id', 'path_to_img'
    
    # Delete columns 'path_to_img' and 'object_id'
    df = df.drop(columns=['object_id', 'path_to_img'])
    
    # Make a stratified sampling by classif_id: 70% of data for training, 15% for validation and 15% for test
    y = df.pop('classif_id')
    X = df
    # 70% of data for training
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
    # split remaining 30% to 50-50 for validation and testing
    X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=0.5, random_state=random_state, stratify=y_eval)
    
    # Sort and reset indexes
    X_train = X_train.sort_index().reset_index(drop=True)
    y_train = y_train.sort_index().reset_index(drop=True)
    X_val   = X_val.sort_index().reset_index(drop=True)
    y_val   = y_val.sort_index().reset_index(drop=True)
    X_test  = X_test.sort_index().reset_index(drop=True)
    y_test  = y_test.sort_index().reset_index(drop=True)
    
    ## Standardize feature values between 0 and 1
    min_max_scaler = MinMaxScaler()
    # Initiate scaler with training data and scale training data
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    
    # Scale validation data
    X_val_scaled = min_max_scaler.transform(X_val)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    
    # Scale testing data
    X_test_scaled = min_max_scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # put back together X and y for training, validation and test dataframes
    df_train = X_train_scaled.copy()
    df_train['classif_id'] = y_train
    df_train = df_train.sort_values('classif_id', axis=0).reset_index(drop=True)

    df_val = X_val_scaled.copy()
    df_val['classif_id'] = y_val
    df_val = df_val.sort_values('classif_id', axis=0).reset_index(drop=True)
    
    df_test = X_test_scaled.copy()
    df_test['classif_id'] = y_test
    df_test = df_test.sort_values('classif_id', axis=0).reset_index(drop=True)
     
    return df_train, df_val, df_test

