import os
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from tensorflow.keras import utils
import lycon
import random
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa


def read_data_cnn(path, frac=1, random_state=None):
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
        df_classes (DataFrame): classes with their living attribute
    """
    
    # Read CSV file
    df = pd.read_csv(path).rename(columns = {'classif_id_1':'classif_id'})
    
    # Extract classes ('classif_id_1' for model training and 'classif_id_2' for posterior ecological groupings) and ecological relevance
    df_classes = df[['classif_id', 'classif_id_2', 'eco_rev']].drop_duplicates().sort_values('classif_id').reset_index(drop=True)
    
    # The classifier is a CNN, keep 'classif_id_1', 'path_to_img' and 'set' split
    df = df[['path_to_img', 'classif_id', 'set']]
    
    # Fraction subsample 
    if frac < 1:
        df = df.groupby(['classif_id','set'], group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=random_state))
        
    
    # Extract training, validation and test splits
    df_train = df[df['set'] == 'train'].drop('set', axis = 1).reset_index(drop=True)
    df_valid = df[df['set'] == 'valid'].drop('set', axis = 1).reset_index(drop=True)
    df_test  = df[df['set'] == 'test'].drop('set', axis = 1).reset_index(drop=True)
    
    # Compute dataset composition
    df_comp = df.groupby(['classif_id','set']).size().unstack(fill_value=0)

    return df_train, df_valid, df_test, df_classes, df_comp


## Define a data generator 
class DataGenerator(utils.Sequence):
    """
    Generate batches of data for CNN.
    
    Args:
        df (DataFrame): dataframe with path to images and classif_id
        classes (list, array): name of classes
        data_dir (str): directory containing data
        batch_size (int): number of images per batch
        image_dimensions (tuple): images dimensions for CNN input
        shuffle (bool): whether to shuffle data
        augment (bool): whether to augment (zoom in/out, flip, shear) data
        px_del (int): number of pixels to delete at bottom of images (e.g. to remove a scale bar)
        preserve_size (bool): whether to preserve size of small images.
            If False, all image are rescaled. If True, large images are rescaled and small images are padded to CNN input dimension.
        random_state (int or RandomState): controls the randomness
    
    Returns:
        A batch of `batch_size` images (4D ndarray) and one-hot encoded labels (2D ndarray)
    """
    
    def __init__(self, df, classes, data_dir, batch_size=32, image_dimensions = (224, 224, 3), shuffle=True, augment=False, px_del = 0, preserve_size=False, random_state=None):
        self.df               = df  
        self.classes          = classes
        self.data_dir         = data_dir            
        self.dim              = image_dimensions    
        self.batch_size       = batch_size          
        self.image_dimensions = image_dimensions    
        self.shuffle          = shuffle             
        self.augment          = augment             
        self.px_del           = px_del               
        self.preserve_size    = preserve_size   
        self.random_state     = random_state
        
        # initialise the one-hot encoder
        mlb = MultiLabelBinarizer(classes=classes)
        self.class_encoder = mlb
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        # shuffle data if chosen
        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
             
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
        'Define a data augmenter which doses horizontalf flip (50% chance), vertical flip (50% chance), zoom and shear'
        
        if self.random_state is not None:
            ia.seed(self.random_state) # set seed for randomness
        
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 20% of all images
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    shear=(-15, 15),  # shear by -15 to +15 degrees
                    mode='edge', # pad images with border picels
                ),
            ],
            random_order=True # apply these transformations in random order
        )
        return seq(images=images)

    def __getitem__(self, index):
        'Generate one batch of data'
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # select data and load images
        paths = [os.path.join(self.data_dir, self.df.path_to_img[k]) for k in indexes]
        images = [lycon.load(p)/255 for p in paths]
        
        batch_prepared_images = []
        output_size = self.image_dimensions[0]
        # resize images to proper dimension
        for img in images:
            h,w = img.shape[0:2]
            
            # delete scale bar of px_del px at bottom of image
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
            
            # create a square, empty output, of desired dimension, filled with padding value
            pad_value = self.get_padding_value(img)
            img_square = np.full(self.image_dimensions, pad_value)
            
            # compute number of pixels to leave blank 
            offset_ver = int((output_size-h)/2) # on top and bottom of image
            offset_hor = int((output_size-w)/2) # on left and right of image
            
            # replace pixels in output by input image
            img_square[offset_ver:offset_ver+img.shape[0], offset_hor:offset_hor+img.shape[1]] = img
            batch_prepared_images.append(img_square)
        
        # convert to array of images        
        batch_prepared_images = np.array([img for img in batch_prepared_images], dtype='float32')
        
        # data augmentation
        if self.augment == True:
            batch_prepared_images = self.augmenter(batch_prepared_images)
            
        ## Labels
        batch_labels = [self.df.classif_id[i] for i in indexes]
        batch_encoded_labels = self.class_encoder.fit_transform([[l] for l in batch_labels])
        
        # Return reshaped images with labels
        return batch_prepared_images, batch_encoded_labels
        
        
def batch_glimpse(batches, classes, n=1):
    """
    Randomly select an image from a batch and display it with its label
    
    Args:
        batches (DataGenerator): data generator to glimpse at
        classes (array): array of taxonomic classes
        n(int): numer of images to look at
    
    Returns:
        nothing
        
    """
    
    for _ in range(n):
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
        df_classes (DataFrame): classes with their living attribute
    """
    
    # Read CSV file
    df = pd.read_csv(path)
    
    # TODO check that mandatory columns are present: 'classif_id', 'living', 'path_to_img'
    
    # Extract living attribute
    df_classes = df[['classif_id', 'living']].drop_duplicates().sort_values('classif_id').reset_index(drop=True)
    
    # Delete columns 'path_to_img' 
    df = df.drop(columns=['path_to_img', 'living'])
    
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
     
    return df_train, df_val, df_test, df_classes

