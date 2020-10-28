import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def gridsearch_rf(df, max_features, min_samples_leaf):
    """
    Do a grid search to find best hyperparameters for random forest model.
    
    Args:
        df (DataFrame): data to use for grid search
        max_features (list): number of variables per node; default sqrt(nb of vars)
        min_samples_leaf (list): min number of objects in leaf; default for classif = 5
    
    Returns:
        cv_res (DataFrame): results of grid search
    """
    
    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Split data and labels
    y = df.pop('classif_id')
    X = df

    # Initiate a RF model
    rf = RandomForestClassifier(n_estimators=200, criterion='gini', min_samples_split=2)
    # NB: set a small min_sample_split to make sure it is min_sample_leaf which determines the depth of the tree.
    
    # Build grid of hyperparameters to explore
    grid = {
        'max_features': max_features, 
        'min_samples_leaf': min_samples_leaf,
    }
    
    # Do grid search
    mcv = GridSearchCV(rf, param_grid=grid, cv=4, n_jobs=12, scoring='accuracy')
    mcv.fit(X=X, y=y)
    
    # Extract results
    cv_res = {
        'max_features':     mcv.cv_results_['param_max_features'], 
        'min_samples_leaf': mcv.cv_results_['param_min_samples_leaf'],
        'mean_valid_accur': mcv.cv_results_['mean_test_score'],
        'std_valid_accur':  mcv.cv_results_['std_test_score'],
    }
    # Convert to datfarame
    cv_res = pd.DataFrame(cv_res)
    

    
    return cv_res
