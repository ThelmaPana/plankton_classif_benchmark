# plankton_classif_benchmark
Benchmark for plankton images classifications methods for images from multiple plankton imaging devices (ISIIS, Zooscan, Flowcam, etc.)

This tool allows you to do a comparison between a Convolutional Neural Network and a Random Forest classifier on a dataset of plankton images. 

## Data
### Instruments
The comparison is to be done on data from multiple plankton imaging devices:
- ISIIS (*In Situ* Ichthyoplankton Imaging System)
- zooscan
- flowcam
- IFCB (Imaging FlowCytobot)
- UVP (Underwater Vision Profiler)

### Input data
Store your input data in `data/<instrument_name>`. Your data must contain an `images` folder with your images, as well as a csv file named `<instrument>_data.csv` with one row per object.
This csv file should contain the following columns:
- `path_to_img`: path to image
- `classif_id`: object classification
- `features_1` to `features_n`: object features for random forest fit (choices for names of these columns are up to you)

It is strongly recommended that each class contain at least 100 images. 

Data will be split into training, validation and testing sets.

## Classification models
### Convolutional Neural Network
A convolutional neural network takes an image as input and predicts a class for this image. 

The CNN backbone is a MobileNetV2 feature extractor (https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4) with depth multiplier of 1.
A classification head with the number of classes to predict is added on top of the backbone. Intermediate fully connected layers with customizable dropout rate can be inserted between both.  

Input images are expected to have color values in the range [0,1] and a size of 224 x 224 pixels. If need be, images are automatically resized by the CNN DataGenerator.

### Random Forest
A random forest takes a vector of features as input and predicts a class from these values.

## Settings
Settings can be customized in the `settings.yaml` file. 
Reproductible results can be obtained using the `random_state` argument.

## Training
Training is done in two phases:
- model is optimized by training on the training set and evaluating on the validation set
- optimized model is trained on the training set and evaluated on the test set never used before

### CNN training
For each step (i.e. epoch) in the training of the CNN model, the model is trained on training data and evaluated on validation data. It is recommended to train for a large number of epochs and later decide where to stop based on the evolution of accuracy and loss for validation data. This process, called early stopping, is implemented in this tool: for each epoch, weights are saved if and only if the results of this epoch are better than previous one. Last saved weights are then used to test the model on the test data.

### RF training
Random Forest parameters are optimized with a gridsearch including:
- number of trees
- number of features to use to compute each split (default for classification is `sqrt(n_features)`)
- minimum number of samples required to be at a leaf node (default for classification is `5`)

For each set of parameters, model is trained on training data and evaluated on validation data. Finally, the best model is trained on training data and tested on test data.

## Outputs
When you run `train_cnn.py` or `train_rf.py`, an output directory is created and results are stored in this directory. Use the notebooks `inspect_results_cnn.ipynb`and `inspect_results_rf.ipynb` to explore the results. 

## TODOs

- [ ] fill the .yaml settings file
- [ ] check values for settings
- [ ] limit number of objects from each class in training set
- [ ] implement the use of class weights for CNN models
- [ ] train models with data from other instruments
- [ ] aggregate all comparisons in one file

