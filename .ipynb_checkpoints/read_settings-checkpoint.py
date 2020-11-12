import yaml
import os


def check_global():
    """
    Check global project settings. 
    
    Returns:
        settings(dict): global settings for the project
        
    """
    
    # Open the yaml settings file and extract global settings
    with open('settings.yaml', 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)['global']
    
    ## Random state
    assert (settings['random_state'] is None) or isinstance(settings['random_state'], int), \
        'global > random_state should be none or an integer'
    
    ## Input data
    # input directory
    assert isinstance(settings['input_data']['instrument'], str), \
        'global > instrument should be a string'
    assert os.path.isdir(os.path.join('data', settings['input_data']['instrument'])), \
        'no data directory found for this instrument'
    
    # train/validation/test split
    assert (isinstance(settings['input_data']['split'], list)) and \
        (sum([isinstance(x, (int, float)) for x in settings['input_data']['split']]) == 3), \
        'global > input_data > split should be a list of 3 numbers'
    assert sum(settings['input_data']['split']) == 100, \
        'global > input_data > split should sum to 100'
    
    # n_max
    assert isinstance(settings['input_data']['n_max'], int) and \
        (settings['input_data']['n_max'] > 0), \
        'global > input_data > n_max should be a positive integer'
    
    return settings

    
def check_rf():
    """
    Check project settings for random forest.
    
    Returns:
        settings(dict): random forest settings for the project
        
    """
    
    # Open the yaml settings file and extract random forest settings
    with open('settings.yaml', 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)['rf']
    
    # n_jobs
    assert isinstance(settings['n_jobs'], int), \
        'rf > n_jobs should be an integer'
    
    # gridsearch
    assert isinstance(settings['grid_search']['go'], bool), \
        'rf > grid_search > go should be a boolean'
    # if a gridsearch is done, values must be provided for hyperparameters to try
    if settings['grid_search']['go']:
        assert (isinstance(settings['grid_search']['max_features_try'], list)) and \
            (sum([isinstance(x, (int, float)) for x in settings['grid_search']['max_features_try']]) == \
             len(settings['grid_search']['max_features_try'])),\
        'rf > grid_search > max_features_try should be a list of numbers'
        assert (isinstance(settings['grid_search']['min_samples_leaf_try'], list)) and \
            (sum([isinstance(x, (int, float)) for x in settings['grid_search']['min_samples_leaf_try']]) == \
             len(settings['grid_search']['min_samples_leaf_try'])),\
        'rf > grid_search > min_samples_leaf_try should be a list of numbers'
        assert (isinstance(settings['grid_search']['n_estimators_try'], list)) and \
            (sum([isinstance(x, int) for x in settings['grid_search']['n_estimators_try']]) == \
             len(settings['grid_search']['n_estimators_try'])),\
        'rf > grid_search > n_estimators_try should be a list of integers'
    # if no gridsearch is done, values must be provided for hyperparameters
    else:
        assert isinstance(settings['hyperparameters']['max_features'], (int, float)), \
            'rf > hyperparameters > max_features should be a number'
        assert isinstance(settings['hyperparameters']['min_samples_leaf'], (int, float)), \
            'rf > hyperparameters > min_samples_leaf should be a number'
        assert isinstance(settings['hyperparameters']['n_estimators'], int), \
            'rf > hyperparameters > min_samples_leaf should be an integer'
        
    return settings

def check_cnn():
    """
    Check project settings for convolutional neural network.
    
    Returns:
        settings(dict): cnn settings for the project
        
    """
     
    # Open the yaml settings file and extract cnn settings
    with open('settings.yaml', 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)['cnn']
    
    # data
    assert isinstance(settings['data']['batch_size'], int) and \
        (settings['data']['batch_size'] > 0), \
        'cnn > data > batch_size should be a positive integer'
    assert isinstance(settings['data']['px_del'], int) and \
        (settings['data']['px_del'] >= 0), \
        'cnn > data > px_del should be zero or a positive integer'
    assert isinstance(settings['data']['preserve_size'], bool), \
        'cnn > data > preserve_size should be a boolean'
    assert isinstance(settings['data']['augment'], bool), \
        'cnn > data > augment should be a boolean'
    assert isinstance(settings['data']['use_weights'], bool), \
        'cnn > data > use_weights should be a boolean'
    
    # architecture
    assert isinstance(settings['architecture']['fc_layers_nb'], int) and \
        (settings['architecture']['fc_layers_nb'] > 0), \
        'cnn > architecture > fc_layers_nb should be a positive integer'
    assert isinstance(settings['architecture']['fc_layers_size'], int) and \
        (settings['architecture']['fc_layers_size'] > 0), \
        'cnn > architecture > fc_layers_size should be a positive integer'
    assert isinstance(settings['architecture']['fc_layers_dropout'], float) and \
        (settings['architecture']['fc_layers_dropout'] >= 0) and \
        (settings['architecture']['fc_layers_dropout'] < 1), \
        'cnn > architecture > fc_layers_dropout should be a float between 0 and 1'
    assert isinstance(settings['architecture']['classif_layer_dropout'], float) and \
        (settings['architecture']['classif_layer_dropout'] >= 0) and \
        (settings['architecture']['classif_layer_dropout'] < 1), \
        'cnn > architecture > classif_layer_dropout should be a float between 0 and 1'
    assert isinstance(settings['architecture']['train_fe'], bool), \
        'cnn > architecture > train_fe should be a boolean'
    
    # compilation
    assert isinstance(settings['compilation']['lr_method'], str) and \
        settings['compilation']['lr_method'] in ['decay', 'constant'], \
        'cnn > compilation > lr_method should be "decay" on "constant"'
    assert isinstance(settings['compilation']['initial_lr'], float) and \
        (settings['compilation']['initial_lr'] > 0) and \
        (settings['compilation']['initial_lr'] < 1), \
        'cnn > compilation > initial_lr should be an float between 0 and 1'
    if settings['compilation']['lr_method'] == 'decay': 
        assert isinstance(settings['compilation']['decay_rate'], float) and \
            (settings['compilation']['decay_rate'] > 0) and \
            (settings['compilation']['decay_rate'] < 1), \
            'cnn > compilation > decay_rate should be an float between 0 and 1'
    assert isinstance(settings['compilation']['loss'], str)and \
        settings['compilation']['loss'] in ['cce', 'sfce'], \
        'cnn > compilation > loss should be "cce" or "sfce"'
    
    # training
    assert (isinstance(settings['training']['epochs'], int)) and \
        (settings['training']['epochs'] > 0), \
        'cnn > training > epochs should be a positive integer'

    return settings
    

    
    