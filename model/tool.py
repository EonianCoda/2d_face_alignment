from model.Regression import RegressionModel

def get_FAN(num_HG:int = 4, HG_depth:int = 4, num_feats:int = 256, backbone:str="FAN"):
    if backbone == "FAN":
        from model.FAN import FAN
    elif backbone == "FAN_SE":
        from model.FAN_SE import FAN
    elif backbone == "FAN_SE2":
        from model.FAN_SE2 import FAN
    elif backbone == "FAN_IR":
        from model.FAN_IR import FAN

    return FAN(num_HG, HG_depth, num_feats)
    
def get_model(cfg:dict):
    model_type = cfg['model_type']
    if model_type == "classifier":
        num_HG = cfg['num_HG']
        HG_depth = cfg['HG_depth']
        num_feats = cfg['num_feats']
        backbone = cfg['backbone'][cfg['backbone_idx']]
        return get_FAN(num_HG, HG_depth, num_feats, backbone)
    elif model_type == "regressor":
        backbone = cfg['backbone'][cfg['backbone_idx']]
        dropout = cfg['dropout']
        return RegressionModel(backbone, dropout=dropout)