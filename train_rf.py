import os
import glob
import datasets
import model_rf
from plotnine import *
from sklearn.metrics import accuracy_score

#################################### Settings ####################################
output_dir = 'rf_output'
# Check if output_directory exists, if not create it
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
delete_previous = True # whether to delete files in output directory
if delete_previous:
    files = glob.glob(os.path.join(output_dir, '*'))
    for f in files:
        os.remove(f)
        
random_state = 42

## Data
instrument = 'isiis'
data_dir = os.path.join('data', instrument)

## RF model
n_jobs = 6 
gridsearch_go = True
max_features_try = [4,6,8,10]
min_samples_leaf_try = [2,5,10]
n_estimators_try = [100,200,350,500]


##################################################################################

## Read data for RF
df_train, df_valid, df_test = datasets.read_data_rf(
    path = os.path.join(data_dir, 'isiis_data.csv'),
    random_state=random_state)


## Grid search
# Do grid serach
if gridsearch_go:
    gs_results, best_params = model_rf.gridsearch_rf(
        df_train, 
        df_valid, 
        max_features_try=max_features_try, 
        min_samples_leaf_try=min_samples_leaf_try, 
        n_estimators_try=n_estimators_try,
        output_dir=output_dir,
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    # Plot results
    ggplot.draw(ggplot(gs_results) +
      geom_point(aes(x='max_features', y='valid_accuracy', colour='factor(n_estimators)'))+
      facet_wrap('~min_samples_leaf', labeller = 'label_both') +
      labs(colour='n_estimators', title = 'Gridsearch results'))
    
    # Set parameters for future RF models
    n_estimators = best_params['n_estimators']
    max_features = best_params['max_features']
    min_samples_leaf = best_params['min_samples_leaf']
    print(f'Selected parameters are: n_estimators = {n_estimators}, max_features = {max_features}, min_samples_leaf = {min_samples_leaf}')


## Fit the RF of training data
# 200 trees is enough
rf = model_rf.train_rf(
    df=df_train, 
    n_estimators=n_estimators, 
    max_features=max_features, 
    min_samples_leaf=min_samples_leaf, 
    n_jobs=n_jobs, 
    random_state=random_state
)


## Evaluate the RF on test data
test_accuracy = model_rf.predict_evaluate_rf(
    rf_model=rf, 
    df=df_test,
    output_dir = output_dir
)


