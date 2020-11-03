import os
import glob
import read_settings
import datasets
import model_rf
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
output_dir = '_'.join(['output_rf', instrument])
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if global_settings['delete_previous']:
    files = glob.glob(os.path.join(output_dir, '*'))
    for f in files:
        os.remove(f)

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
df_train, df_valid, df_test = datasets.read_data_rf(
    path=os.path.join(data_dir, '_'.join([instrument, 'data.csv'])),
    split=split,
    n_max=n_max,
    random_state=random_state)

# Write train, valid and test splits to output directory for future inspection
df_train.to_csv(os.path.join(output_dir, 'df_train.csv'), index=False)
df_valid.to_csv(os.path.join(output_dir, 'df_valid.csv'), index=False)
df_test.to_csv(os.path.join(output_dir, 'df_test.csv'), index=False)


## Grid search
# Do grid serach
if rf_settings['grid_search']['go']:
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


