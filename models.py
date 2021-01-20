import pandas as pd
import numpy as np
import itertools as it
import os
import pickle
import glob

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
from tensorflow.keras import layers, optimizers, losses, callbacks 
import tensorflow_addons as tfa


def gridsearch_rf(df1, df2, max_features_try, min_samples_leaf_try, n_estimators_try, output_dir, n_jobs, class_weights=None, random_state=None):
    """
    Do a grid search to find best hyperparameters for random forest model, including number of estimators.
    
    Args:
        df1 (DataFrame): training data to use to fit grid search
        df2 (DataFrame): validation data to use to evaluate grid search        
        max_features (list): number of variables per node; default sqrt(nb of vars)
        min_samples_leaf (list): min number of objects in leaf; default for classif = 5
        n_estimators (list): number of estimators (usually between 100 and 500)
        output_dir (str): directory where to save gridsearch results
        n_jobs (int): number of cores to use 
        class_weights(dict): weights for classes
        random_state (int or RandomState): controls both the randomness of the bootstrapping and features sampling; default=None
    
    Returns:
        cv_res (DataFrame): results of grid search
        best_params (dict): best parameters based on validation accuracy value
    """
    
    # Shuffle data
    df1 = df1.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df2 = df2.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split data and labels
    y_train = df1['classif_id']
    X_train = df1.drop('classif_id', axis=1)
    
    y_valid = df2['classif_id']
    X_valid = df2.drop('classif_id', axis=1)
    
    # Build grid of hyperparameters to explore
    grid = {
        'max_features': max_features_try, 
        'min_samples_leaf': min_samples_leaf_try,
    }
    # Make a list of all parameters combinations
    keys = list(grid.keys())
    grid_list = list(it.product(*(grid[key] for key in keys)))

    # Initiate empty dict for results
    results = {
        'n_estimators': [],
        'max_features': [],
        'min_samples_leaf': [],
        'valid_accuracy': []
    }

    # First loop on parameters other than n_estimators
    for max_features, min_samples_leaf in grid_list:
        print(f"Trying parameters max_features = {max_features} and min_samples_leaf = {min_samples_leaf}.")
    
        # Initiate a RF model with warm start
        rf = RandomForestClassifier(
            criterion='gini', 
            min_samples_split=2, 
            max_features=max_features, 
            min_samples_leaf=min_samples_leaf,
            warm_start=True,
            n_jobs=n_jobs,
            class_weight=class_weights,
            random_state=random_state
        )
        
        # Second loop on n_estimators
        for n_estimators in n_estimators_try:
            print(f"Number of estimators = {n_estimators}")
            
            # Set number of estimators in RF model
            rf.n_estimators = n_estimators
            
            # Fit on training data
            rf.fit(X=X_train, y=y_train)
            
            # Compute accuracy on validation data
            valid_accuracy = accuracy_score(y_valid, rf.predict(X_valid))
            
            # Store results in dict
            results['n_estimators'].append(n_estimators)
            results['max_features'].append(max_features)
            results['min_samples_leaf'].append(min_samples_leaf)
            results['valid_accuracy'].append(valid_accuracy)

    # Write training history 
    with open(os.path.join(output_dir, 'train_results.pickle'),'wb') as results_file:
        pickle.dump(results, results_file)
        
    # Convert to datfarame
    results = pd.DataFrame(results)
    
    # Extract best parameters based on validation accuracy value
    best_params = results.nlargest(1, 'valid_accuracy').reset_index(drop=True).drop('valid_accuracy', axis=1)
    best_params = best_params.iloc[0].to_dict()
    
    return results, best_params


def train_rf(df, n_estimators, max_features, min_samples_leaf, n_jobs, class_weights, random_state=None):
    """
    Fit a random forest model on data.
    
    Args:
        df (DataFrame): data to use for training
        tree_nb (int): number of trees for the RF model
        max_features (int): number of variables per node
        min_samples_leaf (int): min number of objects in leaf
        n_jobs (int): number of cores to use 
        class_weights(dict): weights for classes
        random_state (int or RandomState): controls both the randomness of the bootstrapping and features sampling; default=None
    
    Returns:
        rf (RandomForestClassifier): fitted random forest model
        
    """
    
    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Split data and labels
    y_train = df['classif_id']
    X_train = df.drop('classif_id', axis=1)
    
    # Initiate RF model
    rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        criterion='gini', 
        min_samples_split=2, 
        min_samples_leaf=min_samples_leaf, 
        max_features=max_features,
        n_jobs=n_jobs,
        class_weight=class_weights,
        random_state=random_state
    )
    
    # Fit the RF model
    rf = rf.fit(X=X_train, y=y_train)
    
    return rf


def predict_evaluate_rf(rf_model, df, df_classes, output_dir):
    """
    Evaluate a random forest model.
    
    Args:
        rf_model (RandomForestClassifier): random forest model to evaluate
        df (DataFrame): data to use for model evaluation
        df_classes (DataFrame): dataframe of classes with living attribute
        output_dir (str): directory where to save prediction results
    
    Returns:
        accuracy (float): accuracy value
    """
    
    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Split data and labels
    y = df['classif_id']
    X = df.drop('classif_id', axis=1)
    
    # Make a list of classes
    classes = df_classes['classif_id'].tolist()
    classes.sort()
    classes = np.array(classes)
    
    # Make a list of living classes
    living_classes = df_classes[df_classes['living']]['classif_id'].tolist()
    living_classes = np.array(living_classes)
    
    # Predict test data
    y_pred = rf_model.predict(X)
    
    # Compute accuracy between true labels and predicted labels
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    living_precision = living_precision_score(y, y_pred, living_classes)
    living_recall = living_recall_score(y, y_pred, living_classes)
    
    print(f'Test accuracy = {accuracy}')
    print(f'Balanced test accuracy = {balanced_accuracy}')
    print(f'Living precision = {living_precision}')
    print(f'Living recall = {living_recall}')
    
    # Write true and predicted classes and accuracy to test file
    with open(os.path.join(output_dir, 'test_results.pickle'),'wb') as test_file:
        pickle.dump({'true_classes': y,
                     'predicted_classes': y_pred,
                     'classes': classes,
                     'living_classes': living_classes,
                     'accuracy': accuracy,
                     'balanced_accuracy': balanced_accuracy,
                     'living_precision': living_precision,
                     'living_recall': living_recall},
                    test_file)
        
    return(accuracy)
    

def create_cnn(fc_layers_nb, fc_layers_dropout, fc_layers_size, classif_layer_dropout, classif_layer_size,  train_fe = False, glimpse = True):

    """
    Generates a CNN model. 
    
    Args:
        fc_layers_nb (int): number of fully connected layers 
        fc_layers_size (int): size of fully connected layers 
        fc_layers_dropout (float): dropout of fully connected layers 
        classif_layer_size (int): size of classification layer (i.e. number of classes to predict)
        classif_layer_dropout (float): dropout of classification layer
        train_fe (bool): whether to train the feature extractor (True) or only classification head (False)
        glimpse(bool): whether to show a model summary
    
    Returns:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNN model
        
    """
    
    ## Initiate empty model
    model = tf.keras.Sequential()
    
    ## MobileNet V2 feature extractor
    #fe_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    fe_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    fe_layer = hub.KerasLayer(fe_url, input_shape=(224, 224, 3))
    # Set feature extractor trainability
    fe_layer.trainable=train_fe
    model.add(fe_layer)
    
    ## Fully connected layers
    if fc_layers_nb:
        for i in range(fc_layers_nb):
            if fc_layers_dropout:
                model.add(layers.Dropout(fc_layers_dropout))
            model.add(layers.Dense(fc_layers_size, activation='relu'))
    
    ### Classification layer
    if classif_layer_dropout:
        model.add(layers.Dropout(classif_layer_dropout))
    model.add(layers.Dense(classif_layer_size))
    
    if glimpse:
        model.summary()

    return model


def compile_cnn(model, initial_lr, steps_per_epoch, lr_method='constant', decay_rate=None, loss='cce'):
    """
    Compiles a CNN model. 
    
    Args:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNN model to compile
        lr_method (str): method for learning rate. 'constant' for a constant learning rate, 'decay' for a decay
        initial_lr (float): initial learning rate. If lr_method is 'constant', set learning rate to this value
        steps_per_epochs (int): number of training steps at each epoch. Usually number_of_epochs // batch_size
        decay_rate (float): rate for learning rate decay
        loss (str): method to compute loss. 'cce' for CategoricalCrossentropy (see https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy), 'sfce' for SigmoidFocalCrossEntropy (see https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy), usefull for unbalanced classes

    
    Returns:
        model (tensorflow.python.keras.engine.sequential.Sequential): compiled CNN model
        
    """
    # TODO if lr_method='decay', decay_rate in mandatory

    ## Learning rate
    if lr_method == 'decay':
        lr = optimizers.schedules.InverseTimeDecay(
                    initial_lr, steps_per_epoch, decay_rate, staircase=False, name=None
        )
    else: # Keep constant learning rate
        lr = initial_lr
    
    ## Optimizer: use Adam
    optimizer = optimizers.Adam(learning_rate=lr)
    
    
    ## Loss
    if loss == 'cce':
        loss = losses.CategoricalCrossentropy(from_logits=True,reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    elif loss == 'sfce':
        loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True,reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    
    
    model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics='accuracy'
    )
    
    return model


def train_cnn(model, train_batches, valid_batches, batch_size, epochs, class_weights, output_dir):
    """
    Trains a CNN model. 
    
    Args:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNN model to train
        train_batches
        train_batches
        batch_size (int): size if batches
        epochs (int): number of epochs to train for
        class_weight(dict): weights for classes
        output_dir (str): directory where to save model weights

    
    Returns:
        nothing
        
    """
    #TODO add class weights
    # Set callbacks
    filepath = os.path.join(output_dir, "weights.{epoch:02d}.hdf5")
    cp_callback = callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=True,
        save_freq='epoch',
        verbose=1)

    # Compute number of steps per epochs     
    steps_per_epoch = len(train_batches)//batch_size
    validation_steps = len(valid_batches)//batch_size
    
    # Fit the model.
    history = model.fit(
        train_batches, 
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_batches,
        validation_steps=validation_steps,
        callbacks=[cp_callback],
        class_weight=class_weights
    )
    
    # Write training history 
    with open(os.path.join(output_dir, "train_results.pickle"),"wb") as results_file:
        pickle.dump(history.history, results_file)
    
    return history


def predict_evaluate_cnn(model, batches, df_classes, output_dir):
    """
    Predict batches and evaluate a CNN model by computing accuracy and loss and writting predictions and accuracy into a test file. 
    
    Args:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNN model to eavluate
        batches (datasets.DataGenerator): batches of data to predict
        df_classes (DataFrame): dataframe of classes with living attribute
        output_dir (str): directory where to save prediction results

    
    Returns:
        accuracy (float) accuracy value for test data
        loss (float): loss (categorical cross entropy) value for test data
        
    """

    # Make a list of classes
    classes = df_classes['classif_id'].tolist()
    classes.sort()
    classes = np.array(classes)
    
    # Make a list of living classes
    living_classes = df_classes[df_classes['living']]['classif_id'].tolist()
    living_classes = np.array(living_classes)
    
    # Load last saved weights to CNN model
    saved_weights = glob.glob(os.path.join(output_dir, "*.hdf5"))
    saved_weights.sort()
    model.load_weights(saved_weights[-1])
    
    # Initiate empty lists for predicted and true labels
    predicted_batches = []
    true_batches = []
    
    # Loop over test batches
#    for image_batch, label_batch in batches:
#        # Predict images of batch
#        predicted_batches.extend(model.predict(image_batch))
#        # Extract true labels of batch
#        true_batches.extend(label_batch)
    
    # Loop over test batches
    for i in range(len(batches)+1):
        # Define image batch and label batch
        image_batch = batches[i][0]
        label_batch = batches[i][1]
        
        # if batch is not empty, predict it
        if len(image_batch) > 0:
            # Predict images of batch
            predicted_batches.extend(model.predict(image_batch))
            # Extract true labels of batch
            true_batches.extend(label_batch)
    
    # Convert to class names
    predicted_classes = classes[np.argmax(predicted_batches, axis=1)]
    true_classes = classes[np.argmax(true_batches, axis=1)]
    
    # Compute accuracy, precision and recall for living classes and loss from true labels and predicted labels
    accuracy = accuracy_score(true_classes, predicted_classes)
    balanced_accuracy = balanced_accuracy_score(true_classes, predicted_classes)
    living_recall = living_recall_score(true_classes, predicted_classes, living_classes)
    living_precision = living_precision_score(true_classes, predicted_classes, living_classes)
    
    # Display results
    print(f'Test accuracy = {accuracy}')
    print(f'Balanced test accuracy = {balanced_accuracy}')
    print(f'Living precision = {living_precision}')
    print(f'Living recall = {living_recall}')
    
    # Write true and predicted classes to test file
    with open(os.path.join(output_dir, 'test_results.pickle'),'wb') as test_file:
        pickle.dump({'true_classes': true_classes,
                     'predicted_classes': predicted_classes,
                     'classes': classes,
                     'living_classes': living_classes,
                     'accuracy': accuracy,
                     'balanced_accuracy': balanced_accuracy,
                     'living_precision': living_precision,
                     'living_recall': living_recall},
                    test_file)
        
    return accuracy


def living_recall_score(y_true, y_pred, classes):
    """
    Compute recall score for a set of classes (usually living classes) and ignoring others (non-living)
    
    Args:
        y_true (1d array): true labels
        y_pred (1d array): predicted labels
        classes (1d array): list of classes to consider for recall computation

    Returns:
        bio_recall (float): accuracy computed only on living classes 
        
    """
    # Initiate zero array for matches between true and pred labels
    eq = np.zeros([len(y_true)])
    # Initiate zero array for classes to include 
    included = np.zeros([len(y_true)])
    
    # Loop over predictions
    for i in range(len(y_true)):
        # If true and predicted labels are identical, put a 1 in the match array
        eq[i] = y_true[i] == y_pred[i]
        # If true label is in living classes, put a one in array for classes to include
        included[i] = y_true[i] in classes
    
    # Sum the element-wise multiplication between matches and classes to include and divide it by the number of included cases
    bio_recall = np.sum(np.multiply(eq, included))/sum(included)

    return bio_recall


def living_precision_score(y_true, y_pred, classes):
    """
    Compute precision score for a set of classes (usually living classes) and ignoring others (non-living)
    
    Args:
        y_true (1d array): true labels
        y_pred (1d array): predicted labels
        classes (1d array): list of classes to consider for precision computation

    Returns:
        bio_precision (float): accuracy computed only on living classes 
        
    """
    # Initiate zero array for matches between true and pred labels
    eq = np.zeros([len(y_true)])
    # Initiate zero array for classes to include 
    included = np.zeros([len(y_true)])
    
    # Loop over predictions
    for i in range(len(y_true)):
        # If true and predicted labels are identical, put a 1 in the match array
        eq[i] = y_true[i] == y_pred[i]
        # If predicted label is in living classes, put a one in array for classes to include
        included[i] = y_pred[i] in classes
    
    # Sum the element-wise multiplication between matches and classes to include and divide it by the number of included cases
    bio_precision = np.sum(np.multiply(eq, included))/sum(included)

    return bio_precision