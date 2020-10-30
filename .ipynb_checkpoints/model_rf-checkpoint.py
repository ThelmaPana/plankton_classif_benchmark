import pandas as pd
import numpy as np
import itertools as it
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def gridsearch_rf(df1, df2, max_features_try, min_samples_leaf_try, n_estimators_try):
    """
    Do a grid search to find best hyperparameters for random forest model, including number of estimators.
    
    Args:
        df1 (DataFrame): training data to use to fit grid search
        df2 (DataFrame): validation data to use to evaluate grid search        
        max_features (list): number of variables per node; default sqrt(nb of vars)
        min_samples_leaf (list): min number of objects in leaf; default for classif = 5
        n_estimators (list): number of estimators (usually between 100 and 500)
    
    Returns:
        cv_res (DataFrame): results of grid search
        best_params (dict): best parameters based on validation accuracy value
    """
    
    # Shuffle data
    df1 = df1.sample(frac=1).reset_index(drop=True)
    df2 = df2.sample(frac=1).reset_index(drop=True)
    
    # Split data and labels
    y_train = df1.pop('classif_id')
    X_train = df1
    
    y_valid = df2.pop('classif_id')
    X_valid = df2
    
    # Build grid of hyperparameters to explore
    grid = {
        'max_features': max_features_try, 
        'min_samples_leaf': min_samples_leaf_try,
    }
    # Make a list of all parameters combinations
    keys = list(grid.keys())
    grid_list = list(it.product(*(grid[key] for key in keys)))

    # Initiate empty dict for results
    results = {
        'n_estimators': [],
        'max_features': [],
        'min_samples_leaf': [],
        'valid_accuracy': []
    }

    # First loop on parameters other than n_estimators
    for max_features, min_samples_leaf in grid_list:
        print(f"Trying parameters max_features = {max_features} and min_samples_leaf = {min_samples_leaf}.")
    
        # Initiate a RF model with warm start
        rf = RandomForestClassifier(
            criterion='gini', 
            min_samples_split=2, 
            max_features=max_features, 
            min_samples_leaf=min_samples_leaf,
            warm_start=True,
            n_jobs=6
        )
        
        # Second loop on n_estimators
        for n_estimators in n_estimators_try:
            print(f"Number of estimators = {n_estimators}")
            
            # Set number of estimators in RF model
            rf.n_estimators = n_estimators
            
            # Fit on training data
            rf.fit(X=X_train, y=y_train)
            # Compute accuracy on validation data
            valid_accuracy = accuracy_score(y_valid, rf.predict(X_valid))
            
            # Store results in dict
            results['n_estimators'].append(n_estimators)
            results['max_features'].append(max_features)
            results['min_samples_leaf'].append(min_samples_leaf)
            results['valid_accuracy'].append(valid_accuracy)

    
    # Convert to datfarame
    results = pd.DataFrame(results)
    
    # Extract best parameters based on validation accuracy value
    best_params = results.nlargest(1, 'valid_accuracy').reset_index(drop=True).drop('valid_accuracy', axis=1)
    best_params = best_params.iloc[0].to_dict()

    
    return results, best_params


def train_rf(df, n_estimators, max_features, min_samples_leaf):
    """
    Fit a random forest model on data.
    
    Args:
        df (DataFrame): data to use for training
        tree_nb (int): number of trees for the RF model
        max_features (int): number of variables per node
        min_samples_leaf (int): min number of objects in leaf
    
    Returns:
        pred_res (DataFrame): results of accuracy per value of tree numbers
        
    """
    # Split data and labels
    y_train = df.pop('classif_id')
    X_train = df
    
    # Initiate RF model
    rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        criterion='gini', 
        min_samples_split=2, 
        min_samples_leaf=min_samples_leaf, 
        max_features=max_features
    )
    
    # Fit the RF model
    rf = rf.fit(X=X_train, y=y_train)
    
    return rf