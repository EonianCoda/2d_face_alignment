
cfg = {
    'scheduler_type': 0, # 0: ReduceLROnPlateau, 1: Warmup_ReduceLROnPlateau
    'warm_epoch': 1, # If scheduler == 1, then use this arg
    'data_root':'./data',
    'train_annot':'./data/train_annot.pkl',
    'split_ratio': 0.9,
    'seed': 987,
    'batch_size': 8,
    'lr': 1e-2,
    'epoch':10,
}