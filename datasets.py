import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from tensorflow.keras import utils
import cv2
import lycon
import random
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa


def read_data_cnn(path, split=[70, 15, 15], n_max=None, random_state=None):
    """
    Read a csv file containing data to train the cnn
    
    Args:
        path (str): path to the file
        split (list): proportions for train, validation and test splits. Sum is 100.
        n_max (NoneType or int): maximum number of objects per class for training set
        random_state (int or RandomState): controls both the randomness of the bootstrapping and features sampling; default=None
    
    Returns:
        df_train (DataFrame): training data containing path to image and classif_id
        df_val (DataFrame): validation data containing path to image and classif_id
        df_test (DataFrame): testing data containing path to image and classif_id
    """
    
    df = pd.read_csv(path)
    
    # TODO check that mandatory columns are present: 'object_id', 'classif_id', 'path_to_img'
    
    # The classifier is a CNN, keep 'classif_id', and 'path_to_img'
    df = df[['path_to_img', 'classif_id']]
    
    # Make a stratified sampling by classif_id: 70% of data for training, 15% for validation and 15% for test
    y = df.pop('classif_id')
    X = df
    # split data for training
    train_split = split[0]/100
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, train_size=train_split, random_state=random_state, stratify=y)
    # split remaining data between validation and testing
    test_size = split[2]/(100-split[0])
    X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=test_size, random_state=random_state, stratify=y_eval)
    
    # put back together X and y for training, validation and test dataframes
    df_train = X_train.copy()
    df_train['classif_id'] = y_train
    df_train = df_train.sort_values('classif_id', axis=0).reset_index(drop=True)

    df_val = X_val.copy()
    df_val['classif_id'] = y_val
    df_val = df_val.sort_values('classif_id', axis=0).reset_index(drop=True)
    
    df_test = X_test.copy()
    df_test['classif_id'] = y_test
    df_test = df_test.sort_values('classif_id', axis=0).reset_index(drop=True)
    
    # Limit number of objects per class for training set
    if n_max:
        df_train = df_train.groupby('classif_id').apply(lambda x: x.sample(min(n_max,len(x)), random_state=random_state)).reset_index(drop=True)
  
    return df_train, df_val, df_test


## Define a data generator 
class DataGenerator(utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, data_dir, batch_size=32, image_dimensions = (224, 224, 3), shuffle=True, augment=False, px_del = 0, preserve_size=False):
        self.df               = df                  # dataframe with path to images and classif_id
        self.data_dir         = data_dir            # directory containing data
        self.dim              = image_dimensions    # image dimensions
        self.batch_size       = batch_size          # batch size
        self.image_dimensions = image_dimensions    # image_dimensions
        self.shuffle          = shuffle             # shuffle bool
        self.augment          = augment             # augment bool
        self.px_del           = px_del              # pixels to delete at bottom of images (scale bar)  
        self.preserve_size    = preserve_size       # preserve_size bool         
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            #np.random.shuffle(self.indexes)
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            
    def class_encoder(self):
        'Encodes classes with multi label binarizer'
        mlb = MultiLabelBinarizer()
        classif_id = self.df['classif_id'].tolist()
        classif_id_enc = mlb.fit_transform([[c] for c in classif_id])
        return classif_id_enc
    
    def get_padding_value(self, img):
        'Compute value to use to pad an image, as the median value of border pixels'
        # get height and width of image
        h = img.shape[0]
        w = img.shape[1]
        
        # concatenate border pixels in an array
        borders = np.concatenate((
            img[:, 0],         # left column
            img[:, w-1],       # right column
            img[0, 1:w-2],     # top line without corners
            img[h-1, 1:w-2],   # bottom line without corners        
        ), axis=0)
        
        # compute the median
        pad_value = np.median(borders)
        
        return pad_value
    
    def augmenter(self, images):
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 20% of all images
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    shear=(-15, 15),  # shear by -15 to +15 degrees
                    cval=(1), # pad images with white
                ),
            ],
            random_order=True
        )
        return seq(images=images)
    


    def __getitem__(self, index):
        'Generate one batch of data'
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # select data and load images
        paths = [os.path.join(self.data_dir, self.df.path_to_img[k]) for k in indexes]
        images = [cv2.imread(p)/255 for p in paths]
        
        square_images = []
        output_size = self.image_dimensions[0]
        # resize images to proper dimension
        for img in images:
            h = img.shape[0]
            w = img.shape[1] 
            
            # delete scale bar of 31px at bottom of image
            img = img[0:h-self.px_del,:]
            h = img.shape[0]
            
            # compute largest dimension (hor or ver)
            dim_max = int(max(h, w))
            
            # if size is not preserved or image is larger than output_size, resize image to output_size
            if not(self.preserve_size) or (dim_max > output_size):
                # Resize image so that largest dim is now equal to output_size
                img = lycon.resize(
                    img, 
                    height = max(h*output_size//max(h,w),1), 
                    width = max(w*output_size//max(h,w),1), 
                    interpolation=lycon.Interpolation.AREA
                )
                h = img.shape[0]
                w = img.shape[1]  
            
            # create a square, blank output, of desired dimension
            #img_square = np.ones(output_shape)
            pad_value = self.get_padding_value(img)
            
            #img_square = np.ones(self.image_dimensions)
            img_square = np.full(self.image_dimensions, pad_value)
            
            # compute number of pixels to leave blank 
            offset_ver = int((output_size-h)/2) # on top and bottom of image
            offset_hor = int((output_size-w)/2) # on left and right of image
            
            # replace pixels in output by input image
            img_square[offset_ver:offset_ver+img.shape[0], offset_hor:offset_hor+img.shape[1]] = img
            square_images.append(img_square)
        
        # convert to array of images        
        square_images = np.array([img for img in square_images], dtype='float32')
        
        # data augmentation
        if self.augment == True:
            square_images = self.augmenter(square_images)
            
        
        ## Labels
        classif_id_enc = self.class_encoder()
        labels = [classif_id_enc[k].tolist() for k in indexes]
        labels = np.array(labels, dtype='float32')
        
        # Return reshaped images with labels
        return square_images, labels
    

def batch_glimpse(batches, classes):
    """
    Randomly select an image from a batch and display it with its label
    
    Args:
        batches (str): 
        classes (array): array of taxonomic classes
    
    Returns:
        nothing
        
    """

    b = random.randint(0, len(batches)-1)
    image_batch, label_batch = batches[b]
    i = random.randint(0, len(label_batch)-1)
    plt.imshow(image_batch[i][:,:,0], cmap='gray')
    plt.title(classes[np.argmax(label_batch[i])])
    plt.show()
    pass


def read_data_rf(path, split = [70, 15, 15], n_max=None, random_state=None):
    """
    Read a csv file containing data to train the RandomForest and scale features between 0 and 1. 
    
    Args:
        path (str): path to the file
        split (list): proportions for train, validation and test splits. Sum is 100.
        n_max (NoneType or int): maximum number of objects per class for training set
        random_state (int or RandomState): controls both the randomness of the bootstrapping and features sampling; default=None
    
    Returns:
        df_train (DataFrame): training data containing object features and classif_id
        df_val (DataFrame): validation data containing object features and classif_id
        df_test (DataFrame): testing data containing object features and classif_id
    """
    
    df = pd.read_csv(path)
    
    # TODO check that mandatory columns are present: 'object_id', 'classif_id', 'path_to_img'
    
    # Delete columns 'path_to_img' 
    df = df.drop(columns=['path_to_img'])
    
    # Make a stratified sampling by classif_id
    y = df.pop('classif_id')
    X = df
    # split data for training
    train_split = split[0]/100
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, train_size=train_split, random_state=random_state, stratify=y)
    # split remaining data between validation and testing
    test_size = split[2]/(100-split[0])
    X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=test_size, random_state=random_state, stratify=y_eval)
    
    # Sort and reset indexes
    X_train = X_train.sort_index().reset_index(drop=True)
    y_train = y_train.sort_index().reset_index(drop=True)
    X_val   = X_val.sort_index().reset_index(drop=True)
    y_val   = y_val.sort_index().reset_index(drop=True)
    X_test  = X_test.sort_index().reset_index(drop=True)
    y_test  = y_test.sort_index().reset_index(drop=True)
    
    ## Standardize feature values between 0 and 1
    min_max_scaler = MinMaxScaler()
    # Initiate scaler with training data and scale training data
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    
    # Scale validation data
    X_val_scaled = min_max_scaler.transform(X_val)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    
    # Scale testing data
    X_test_scaled = min_max_scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # put back together X and y for training, validation and test dataframes
    df_train = X_train_scaled.copy()
    df_train['classif_id'] = y_train
    df_train = df_train.sort_values('classif_id', axis=0).reset_index(drop=True)

    df_val = X_val_scaled.copy()
    df_val['classif_id'] = y_val
    df_val = df_val.sort_values('classif_id', axis=0).reset_index(drop=True)
    
    df_test = X_test_scaled.copy()
    df_test['classif_id'] = y_test
    df_test = df_test.sort_values('classif_id', axis=0).reset_index(drop=True)
    
    # Limit number of objects per class for training set
    if n_max:
        df_train = df_train.groupby('classif_id').apply(lambda x: x.sample(min(n_max,len(x)), random_state=random_state)).reset_index(drop=True)
     
    return df_train, df_val, df_test

