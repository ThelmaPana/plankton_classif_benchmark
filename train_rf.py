import os
import datasets
import model_rf
from plotnine import *

from sklearn.metrics import accuracy_score

#################################### Settings ####################################
output_dir = 'rf_output'
# Check if output_dir exists, if not create it
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
random_state = 42

## Data
instrument = 'isiis'
data_dir = os.path.join('data', instrument)

## Gridsearch
gridsearch_go = True
max_features = [4,6,8,10]
min_samples_leaf = [2,5,10]

## Number of trees
tree_nb_go = True
max_tree_nb = 500

##################################################################################

## Read data for RF
df_train, df_valid, df_test = datasets.read_data_rf(
    path = os.path.join(data_dir, 'isiis_data.csv'),
    random_state=random_state)


## Grid search
if gridsearch_go:
    cv_res, max_features,  min_samples_leaf = model_rf.gridsearch_rf(df_train, max_features, min_samples_leaf)
    ggplot.draw(ggplot(cv_res) +
      geom_point(aes(x='max_features', y='mean_valid_accur', colour='min_samples_leaf')))


## Find appropriate number of trees
if tree_nb_go:
    pred_res = model_rf.explore_tree_nb(df_train, df_valid, max_tree_nb, min_samples_leaf, max_features)
    ggplot.draw(ggplot(pred_res) + geom_path(aes(trees, accur)))
