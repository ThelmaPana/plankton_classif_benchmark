import os
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(path, classifier, random_state=None):
    """
    Read a csv file containing data to train the cnn
    
    Args:
        path (str): path to the file
    
    Returns:
        (DataFrame) with the content of the file
    """
    
    df = pd.read_csv(path)
    
    # TODO check that mandatory columns are present: 'object_id', 'classif_id', 'path_to_img'
    
    # If classifier is a CNN, keep 'classif_id', and 'path_to_img'
    if classifier == "CNN":
        df = df[['path_to_img', 'classif_id']]
        
    # If classifier is a RF, keep, 'classif_id' and other features
    elif classifier == "RF":
        df = df.drop(columns=['object_id', 'path_to_img'])

    
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
    df_train['set'] = 'train'
    df_train = df_train.sort_values('classif_id', axis=0).reset_index(drop=True)

    df_val = X_val.copy()
    df_val['classif_id'] = y_val
    df_val['set'] = 'val'
    df_val = df_val.sort_values('classif_id', axis=0).reset_index(drop=True)
    
    df_test = X_val.copy()
    df_test['classif_id'] = y_test
    df_test['set'] = 'test'
    df_test = df_test.sort_values('classif_id', axis=0).reset_index(drop=True)

    
    return df_train, df_val, df_test
