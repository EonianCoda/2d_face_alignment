
cfg = {
    # Scheduler setting
    # 0: ReduceLROnPlateau, 1: Warmup_ReduceLROnPlateau 
    # If scheduler == 1, then use warm_epoch arg
    'scheduler_type': 1, 
    'warm_epoch': 2, 
    # training data setting
    'train_annot':'./data/train_annot.pkl',
    'train_data_root':'./data/train',
    'split_ratio': 0.9,
    'transform':{'flip':False,
                 'roation':True,
                 'noise':True,},
    # testing data
    'test_annot':'./data/val_annot.pkl',
    'test_data_root':'./data/val',
    # Training hyperparameters
    'seed': 987,
    'batch_size': 8,
    'lr': 1e-3,
    'epoch':20,
}