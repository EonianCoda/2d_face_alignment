cfg = {
    ### Loss type ###  
    'loss_idx': 0,
    'losses': {0:"L2",
                1:"adaptive_wing_loss",
                2:"weighted_L2"},
    'weight': 2, # For weighted L2 and adaptive_wing_loss
    
    ### Scheduler setting ###
    'warm_step': 2000,
    'lr':1e-4,
    'milestones': [],
    'milestones_lr': [],
    ### Optimizer ###
    'weight_decay': 1e-6,
    ### Model arichitecture ###
    'num_HG': 2,
    'HG_depth':4,
    'num_feats':128,
    'fix_coord': True,
    'bg_negative': False,
    # For coordinate conv
    'use_CoordConv': True,
    'with_r': False,
    'add_CoordConv_inHG': False,
    ### Attention Block ###
    'attention_block_idx': 2,
    'attention_blocks': {0: "None",
                        1: "SELayer",
                        2: "CA_Block"},

    ### Augumentation Setting ###
    'aug_setting':{'flip': False,
                'rotation': True,
                'noise': False,
                'gaussianBlur': False,
                'colorJitter': False,
                'padding': False,
                'erasing': False,
                'grayscale': False},
    ### Training hyperparameters ###
    'batch_size':8,
    'update_batch_size': 8,
    'epoch':20,
    ### training setting ##
    'train_annot':'./data/train_annot.pkl',
    'train_data_root':'./data/train',
    'split_ratio': 0.9,
    ### testing data ##
    'test_annot':'./data/val_annot.pkl',
    'test_data_root':'./data/val',
    
    'seed': 815,
}
