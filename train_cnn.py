import os
import datasets

data_dir = 'data/isiis'

classifier = "RF"
random_state = 42

df_train, df_valid, df_test = datasets.read_data(
    path = os.path.join(data_dir, 'isiis_data.csv'), 
    classifier = classifier, 
    random_state=random_state)

