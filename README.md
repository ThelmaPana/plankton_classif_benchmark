# plankton_classif_benchmark
Benchmark for plankton images classifications methods (CNN, RF) for images from multiple plankton imaging devices (ISIIS, Zooscan, Flowcam, etc.)

This tool allows you to do a comparison between a Convolutional Neural Network and a Random Forest Classifier on a dataset of plankton images. 


## Instruments
The comparison is to be done on data from multiple devices:
- ISIIS (In Situ Ichthyoplankton Imaging System)
- zooscan
- flowcam
- IFCB (Imaging FlowCytobot)
- UVP (Underwater Vision Profiler)


## Input data
Store your input data in `data/<instrument_name>`. Your data must contain an `images` folder with your images, as well as a csv file named `<instrument>_data.csv` with one row per object.
This csv file should contain the following columns:
- `path_to_img`: path to image
- `classif_id`: object classification
- `features_1` to `features_n`: object features for random forest fit (choices for names of these columns are up to you)

It is strongly recommanded that each classes constains at least 100. 


## Settings
You can customize the settings in the `settings.yaml` file. 


## Outputs
When you run `train_cnn.py` or `train_rf.py`, an output directory is created and results are stored in this directory. Use the notebooks `inspect_results_cnn.ipynb`and `inspect_results_rf.ipynb` to explore the results. 


## TODOs
- [ ] fill the .yaml settings file
- [ ] check values for settings
- [ ] limit number of objects from each class in training set
- [ ] implement the use of class weights for CNN models
- [ ] train models with data from other instruments

