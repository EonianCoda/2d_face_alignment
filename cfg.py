
cfg = {
    ### Model setting ###
    # model type
    'model_type_idx': 0,
    'model_type': ['classifier','regressor'],
    # If model_type == "classifier", then use this arg 
    'num_HG': 1, 
    # If model_type == "regressor", then use this arg 
    'backbone_idx': 0,
    'backbone': {0: "mobilenet_v2",
                 1: "efficientnet_b0",
                 2: "mobilenet_v3_small"},
    'dropout': 0.2,
    ### Scheduler setting ###
    'scheduler_type': 1,  # 0: ReduceLROnPlateau, 1: Warmup_ReduceLROnPlateau 
    'warm_epoch': 2,   # If scheduler == 1, then use warm_epoch arg
    ### training setting ##
    'train_annot':'./data/train_annot.pkl',
    'train_data_root':'./data/train',
    'split_ratio': 0.9,
    'transform':{'flip':False,
                 'roation':True,
                 'noise':True,},
    ### testing data ##
    'test_annot':'./data/val_annot.pkl',
    'test_data_root':'./data/val',
    ### Training hyperparameters ###
    'seed': 987,
    'batch_size': {'classifier': 8,
                    'regressor': 16},
    'lr': {'classifier': 1e-4,
            'regressor': 1e-3},
    'epoch':10,
}