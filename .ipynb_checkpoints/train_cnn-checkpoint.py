import os
import datasets
import model_cnn


#################################### Settings ####################################
## Data
instrument = "isiis"
data_dir = os.path.join('data', instrument)
random_state = 42

## CNN model
# architecture
fc_layers_nb = 2
fc_layers_size = 1280
fc_layers_dropout = 0.4
classif_layer_size = 22
classif_layer_dropout = 0.2
train_layers = 'all'


#path = os.path.join(data_dir, 'isiis_data.csv')

## Read data for CNN
df_train, df_valid, df_test = datasets.read_data_cnn(
    path = os.path.join(data_dir, 'isiis_data.csv'),
    random_state=random_state)


# Number of plankton classes
nb_classes = df_train['classif_id'].nunique()

## Generate CNN
my_cnn = model_cnn.create_cnn(
    fc_layers_nb,
    fc_layers_dropout, 
    fc_layers_size, 
    classif_layer_dropout, 
    classif_layer_size=nb_classes, 
    train_layers='all', 
    glimpse=True)
