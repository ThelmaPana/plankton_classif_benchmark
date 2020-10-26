import os
import datasets

data_dir = 'data/isiis'

random_state = 42

path = os.path.join(data_dir, 'isiis_data.csv')

df_train, df_valid, df_test = datasets.read_data_cnn(
    path = os.path.join(data_dir, 'isiis_data.csv'),
    random_state=random_state)

df_train, df_valid, df_test = datasets.read_data_rf(
    path = os.path.join(data_dir, 'isiis_data.csv'),
    random_state=random_state)

