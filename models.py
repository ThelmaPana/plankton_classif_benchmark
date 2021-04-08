import pandas as pd
import numpy as np
import itertools as it
import os
import pickle
import glob

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score

import tensorflow as tf
import tensorflow_hub as hub
os.environ['TFHUB_CACHE_DIR'] = '.tf_models'
import tensorflow_addons as tfa
from tensorflow.keras import layers, optimizers, losses, callbacks 
import tensorflow_addons as tfa


def gridsearch_rf(df1, df2, max_features_try, min_samples_leaf_try, n_estimators_try, output_dir, n_jobs, class_weights=None, random_state=None):
    """
    Perform a grid search to find best hyperparameters for random forest model.
    
    Args:
        df1 (DataFrame): training data to use to fit grid search
        df2 (DataFrame): validation data to use to evaluate grid search        
        max_features_try (list): tries for number of variables per node; default sqrt(nb of vars)
        min_samples_leaf_try (list): tries for min number of objects in leaf; default for classif = 5
        n_estimators_try (list): tries for number of estimators (usually between 100 and 500)
        output_dir (str): directory where to save gridsearch results
        n_jobs (int): number of cores to use 
        class_weights (dict): weights for classes
        random_state (int or RandomState): controls both the randomness of the bootstrapping and features sampling; default=None
    
    Returns:
        results (DataFrame): results of grid search
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
        nothing
    """
    
    # Split data and labels
    y = df['classif_id'].tolist()
    y.sort()
    y = np.array(y)
    X = df.drop('classif_id', axis=1)

    # Make a list of classes
    classes = df_classes['classif_id'].tolist()
    classes.sort()
    classes = np.array(classes)
    
    # and of regrouped classes
    classes_g = df_classes['classif_id_2'].tolist()
    classes_g = list(set(classes_g))
    classes_g.sort()
    classes_g = np.array(classes_g)
    
    # Make a list of ecologically relevant classes
    eco_rev_classes = df_classes[df_classes['eco_rev']]['classif_id'].tolist()
    eco_rev_classes = np.array(eco_rev_classes)
    
    # Make a list of ecologically relevant classes for grouped classes
    eco_rev_classes_g = df_classes[df_classes['eco_rev']]['classif_id_2'].tolist()
    eco_rev_classes_g = np.array(eco_rev_classes_g)
    
    # Predict test data
    y_pred = rf_model.predict(X)
    
    # Compute accuracy between true labels and predicted labels
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    eco_rev_precision = precision_score(y, y_pred, labels=eco_rev_classes, average='weighted', zero_division=0)
    eco_rev_recall = recall_score(y, y_pred, labels=eco_rev_classes, average='weighted', zero_division=0)
    
    # Display results
    print(f'Test accuracy = {accuracy}')
    print(f'Balanced test accuracy = {balanced_accuracy}')
    print(f'Weighted ecologically relevant precision = {eco_rev_precision}')
    print(f'Weighted ecologically relevant recall = {eco_rev_recall}')
     
    ## Now do the same after regrouping objects to larger classes
    # Generate taxonomy match between taxo used for classif and larger ecological classes 
    taxo_match = df_classes.set_index('classif_id').to_dict('index')
    
    # Convert true classes to larger ecological classes
    y_g = np.array([taxo_match[t]['classif_id_2'] for t in y])
    
    # Convert predicted classes to larger ecological classes
    y_pred_g = np.array([taxo_match[p]['classif_id_2'] for p in y_pred])
    
    # Compute accuracy, precision and recall for living classes and loss from true labels and predicted labels
    accuracy_g = accuracy_score(y_g, y_pred_g)
    balanced_accuracy_g = balanced_accuracy_score(y_g, y_pred_g)
    eco_rev_precision_g = precision_score(y_g, y_pred_g, labels=eco_rev_classes_g, average='weighted', zero_division=0)
    eco_rev_recall_g = recall_score(y_g, y_pred_g, labels=eco_rev_classes_g, average='weighted', zero_division=0)
    
    # Display results
    print(f'Grouped test accuracy = {accuracy_g}')
    print(f'Grouped balanced test accuracy = {balanced_accuracy_g}')
    print(f'Grouped weighted ecologically relevant precision = {eco_rev_precision_g}')
    print(f'Grouped weighted ecologically relevant recall = {eco_rev_recall_g}')

    # Write classes and test metrics into a test file
    with open(os.path.join(output_dir, 'test_results.pickle'),'wb') as test_file:
        pickle.dump({
            'classes': classes,
            'classes_g': classes_g,
            'eco_rev_classes': eco_rev_classes,
            'eco_rev_classes_g': eco_rev_classes_g,
            'true_classes': y,
            'predicted_classes': y_pred,
            'true_classes_g': y_g,
            'predicted_classes_g': y_pred_g,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'eco_rev_precision': eco_rev_precision,
            'eco_rev_recall': eco_rev_recall,
            'accuracy_g': accuracy_g,
            'balanced_accuracy_g': balanced_accuracy_g,
            'eco_rev_precision_g': eco_rev_precision_g,
            'eco_rev_recall_g': eco_rev_recall_g,
        },
        test_file)

    pass
    

def create_cnn(fc_layers_nb, fc_layers_dropout, fc_layers_size, classif_layer_dropout, classif_layer_size,  train_fe = False, glimpse = True):

    """
    Generates a CNN model. 
    
    Args:
        fc_layers_nb (int): number of fully connected layers 
        fc_layers_dropout (float): dropout of fully connected layers
        fc_layers_size (int): size of fully connected layers 
        classif_layer_dropout (float): dropout of classification layer
        classif_layer_size (int): size of classification layer (i.e. number of classes to predict)
        train_fe (bool): whether to train the feature extractor (True) or only classification head (False)
        glimpse(bool): whether to show a model summary
    
    Returns:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNN model
    """
    
    ## Initiate empty model
    model = tf.keras.Sequential()
    
    ## MobileNet V2 feature extractor
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
    
    ## Classification layer
    if classif_layer_dropout:
        model.add(layers.Dropout(classif_layer_dropout))
    model.add(layers.Dense(classif_layer_size))
    
    if glimpse:
        model.summary()

    return model


def compile_cnn(model, lr_method, initial_lr, steps_per_epoch, decay_rate=None, loss='cce'):
    """
    Compiles a CNN model. 
    
    Args:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNN model to compile
        lr_method (str): method for learning rate. 'constant' for a constant learning rate, 'decay' for a decay
        initial_lr (float): initial learning rate. If lr_method is 'constant', set learning rate to this value
        steps_per_epochs (int): number of training steps at each epoch. Usually number_of_samples // batch_size or len(train_batches)
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


def train_cnn(model, train_batches, valid_batches, batch_size, epochs, class_weights, output_dir, workers):
    """
    Trains a CNN model. 
    
    Args:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNN model to train
        train_batches: batches of training data
        valid_batches: batches of validation data
        batch_size (int): size of batches
        epochs (int): number of epochs to train for
        class_weight(dict): weights for classes
        output_dir (str): directory where to save model weights
        workers (int): number of parallel threads for data generators
    
    Returns:
        nothing
    """
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
    
    # Fit the model.
    history = model.fit(
        train_batches, 
        epochs=epochs,
        validation_data=valid_batches,
        callbacks=[cp_callback],
        class_weight=class_weights,
        max_queue_size=max(10, workers*2),
        workers=workers
    )
    
    # Write training history 
    with open(os.path.join(output_dir, "train_results.pickle"),"wb") as results_file:
        pickle.dump(history.history, results_file)
    
    return history


def predict_evaluate_cnn(model, batches, true_classes, df_classes, output_dir, workers):
    """
    Predict batches and evaluate a CNN model.
    Predict images from test set and compute accuracy, balanced_accuracy, precision and recall on relevant classes.
    Regroup objects into larger ecological classes and re-evaluate model.
    
    Args:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNN model to evaluate
        batches (datasets.DataGenerator): batches of test data to predict
        true_classes (list): true classes of test images
        df_classes (DataFrame): dataframe of classes with living attribute
        output_dir (str): directory where to save prediction results
        workers(int): number of parallel threads for data generators

    Returns:
        Nothing
    """

    # Make a list of classes
    classes = df_classes['classif_id'].tolist()
    classes.sort()
    classes = np.array(classes)
    
    # and of regrouped classes
    classes_g = df_classes['classif_id_2'].tolist()
    classes_g = list(set(classes_g))
    classes_g.sort()
    classes_g = np.array(classes_g)
    
    # Make a list of ecologically relevant classes
    eco_rev_classes = df_classes[df_classes['eco_rev']]['classif_id'].tolist()
    eco_rev_classes = np.array(eco_rev_classes)
    
    # Make a list of ecologically relevant classes for grouped classes
    eco_rev_classes_g = df_classes[df_classes['eco_rev']]['classif_id_2'].tolist()
    eco_rev_classes_g = np.array(eco_rev_classes_g)
    
    # Load last saved weights to CNN model
    saved_weights = glob.glob(os.path.join(output_dir, "*.hdf5"))
    saved_weights.sort()
    model.load_weights(saved_weights[-1])
    
    # Predict test batches and convert predictions to plankton classes
    logits = model.predict(batches, max_queue_size=max(10, workers*2), workers=workers)
    predicted_classes = classes[np.argmax(logits, axis=1)]

    # Compute accuracy, precision and recall for living classes and loss from true labels and predicted labels
    accuracy = accuracy_score(true_classes, predicted_classes)
    balanced_accuracy = balanced_accuracy_score(true_classes, predicted_classes)
    eco_rev_precision = precision_score(true_classes, predicted_classes, labels=eco_rev_classes, average='weighted', zero_division=0)
    eco_rev_recall = recall_score(true_classes, predicted_classes, labels=eco_rev_classes, average='weighted', zero_division=0)
    
    # Display results
    print(f'Test accuracy = {accuracy}')
    print(f'Balanced test accuracy = {balanced_accuracy}')
    print(f'Weighted ecologically relevant precision = {eco_rev_precision}')
    print(f'Weighted ecologically relevant recall = {eco_rev_recall}')
    
    ## Now do the same after regrouping objects to larger classes
    # Generate taxonomy match between taxo used for classif and larger ecological classes 
    taxo_match = df_classes.set_index('classif_id').to_dict('index')
    
    # Convert true classes to larger ecological classes
    true_classes_g = np.array([taxo_match[t]['classif_id_2'] for t in true_classes])
    
    # Convert predicted classes to larger ecological classes
    predicted_classes_g = np.array([taxo_match[p]['classif_id_2'] for p in predicted_classes])
    
    # Compute accuracy, precision and recall for living classes and loss from true labels and predicted labels
    accuracy_g = accuracy_score(true_classes_g, predicted_classes_g)
    balanced_accuracy_g = balanced_accuracy_score(true_classes_g, predicted_classes_g)
    eco_rev_precision_g = precision_score(true_classes_g, predicted_classes_g, labels=eco_rev_classes_g, average='weighted', zero_division=0)
    eco_rev_recall_g = recall_score(true_classes_g, predicted_classes_g, labels=eco_rev_classes_g, average='weighted', zero_division=0)
    
    # Display results
    print(f'Grouped test accuracy = {accuracy_g}')
    print(f'Grouped balanced test accuracy = {balanced_accuracy_g}')
    print(f'Grouped weighted ecologically relevant precision = {eco_rev_precision_g}')
    print(f'Grouped weighted ecologically relevant recall = {eco_rev_recall_g}')
    
    # Write classes and test metrics into a test file
    with open(os.path.join(output_dir, 'test_results.pickle'),'wb') as test_file:
        pickle.dump({
            'classes': classes,
            'classes_g': classes_g,
            'eco_rev_classes': eco_rev_classes,
            'eco_rev_classes_g': eco_rev_classes_g,
            'true_classes': true_classes,
            'predicted_classes': predicted_classes,
            'true_classes_g': true_classes_g,
            'predicted_classes_g': predicted_classes_g,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'eco_rev_precision': eco_rev_precision,
            'eco_rev_recall': eco_rev_recall,
            'accuracy_g': accuracy_g,
            'balanced_accuracy_g': balanced_accuracy_g,
            'eco_rev_precision_g': eco_rev_precision_g,
            'eco_rev_recall_g': eco_rev_recall_g,
        },
        test_file)

    pass
