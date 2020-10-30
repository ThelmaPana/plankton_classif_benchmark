import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def gridsearch_rf(df1, df2, max_features, min_samples_leaf):
    """
    Do a grid search to find best hyperparameters for random forest model.
    
    Args:
        df1 (DataFrame): training data to use to fit grid search
        df2 (DataFrame): validation data to use to evaluate grid search        
        max_features (list): number of variables per node; default sqrt(nb of vars)
        min_samples_leaf (list): min number of objects in leaf; default for classif = 5
    
    Returns:
        cv_res (DataFrame): results of grid search
        best_max_features (int): best value for number of variables per node
        best_min_samples_leaf (int): best value for min number of objects in leaf
    """
    
    # Shuffle data
    df1 = df1.sample(frac=1).reset_index(drop=True)
    df2 = df2.sample(frac=1).reset_index(drop=True)
    
    # Split data and labels
    y_train = df1.pop('classif_id')
    X_train = df1
    
    y_valid = df2.pop('classif_id')
    X_valid = df2
    
    # Concat training and validation data
    X = pd.concat([X_train, X_valid]).reset_index(drop=True)
    y = pd.concat([y_train, y_valid]).reset_index(drop=True)
    
    # Set training and testing indices
    train_indices = np.full((len(y_train),), -1, dtype=int)
    test_indices  = np.full((len(y_valid),),  0, dtype=int)
    test_fold = np.append(train_indices, test_indices)
    ps = PredefinedSplit(test_fold)
    
    # Initiate a RF model
    rf = RandomForestClassifier(n_estimators=200, criterion='gini', min_samples_split=2)
    # NB: set a small min_sample_split to make sure it is min_sample_leaf which determines the depth of the tree.
    
    # Build grid of hyperparameters to explore
    grid = {
        'max_features': max_features, 
        'min_samples_leaf': min_samples_leaf,
    }
    
    # Do grid search
    mcv = GridSearchCV(rf, param_grid=grid, cv=ps, n_jobs=12, scoring='accuracy')
    mcv.fit(X=X, y=y)
    
    # Extract results
    cv_res = {
        'max_features':     mcv.cv_results_['param_max_features'], 
        'min_samples_leaf': mcv.cv_results_['param_min_samples_leaf'],
        'mean_valid_accur': mcv.cv_results_['mean_test_score'],
    }
    
    # Convert to datfarame
    cv_res = pd.DataFrame(cv_res)
    
    # Extract best values for max_features and min_samples_leaf
    best_max_features = cv_res.nlargest(1, 'mean_valid_accur').reset_index(drop=True).loc[0, 'max_features']
    best_min_samples_leaf = cv_res.nlargest(1, 'mean_valid_accur').reset_index(drop=True).loc[0, 'min_samples_leaf']

    
    return cv_res, best_max_features, best_min_samples_leaf


def explore_tree_nb(d_train, d_valid, max_tree_nb, min_samples_leaf, max_features):
    """
    Compute accuracy of a random forest model for multiple number of trees. 
    
    Args:
        d_train (DataFrame): data to use for training
        d_valid (DataFrame): data to use to compute accuracy
        max_tree_nb (int): maximum number of trees
        max_features (int): number of variables per node
        min_samples_leaf (int): min number of objects in leaf
    
    Returns:
        pred_res (DataFrame): results of accuracy per value of tree numbers
        
    """
    # Split labels and data
    y_train = d_train.pop('classif_id')
    X_train = d_train
    y_valid = d_valid.pop('classif_id')
    X_valid = d_valid

    # Refit model with best parameters
    rf = RandomForestClassifier(
        n_estimators=0, 
        criterion='gini', 
        min_samples_split=2, 
        min_samples_leaf=min_samples_leaf, 
        max_features=max_features, 
        warm_start=True,
    )

    # Set range of trees to explore
    trees = range(1, max_tree_nb)
    # Initiate empty array for accuracy
    accur = np.full(max_tree_nb-1, np.nan)
    # Loop over number of trees
    for n in trees:
        # Set number of trees
        rf.n_estimators = n
        # Fit training data
        rf = rf.fit(X=X_train, y=y_train)
        # Compute accuracy on validation data
        accur[n-1] = accuracy_score(y_valid, rf.predict(X_valid))

    # Store results in a dataframe
    pred_res = pd.DataFrame.from_dict({'trees': trees, 'accur': accur})
    
    return pred_res


def train_rf(df, tree_nb, max_features, min_samples_leaf):
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
        n_estimators=tree_nb, 
        criterion='gini', 
        min_samples_split=2, 
        min_samples_leaf=min_samples_leaf, 
        max_features=max_features, 
        warm_start=True,
    )
    
    # Fit the RF model
    rf = rf.fit(X=X_train, y=y_train)
    
    return rf