import os
import datasets


#################################### Settings ####################################
output_dir = 'rf_output'
# Check if output_dir exists, if not create it
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
random_state = 42

## Data
instrument = "isiis"
data_dir = os.path.join('data', instrument)

##################################################################################

## Read data for CNN
df_train, df_valid, df_test = datasets.read_data_rf(
    path = os.path.join(data_dir, 'isiis_data.csv'),
    random_state=random_state)
