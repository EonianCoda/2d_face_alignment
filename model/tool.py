from model.Regression import RegressionModel
from model.FAN import FAN
from model.utils import HPM_ConvBlock, Bottleneck, InvertedResidual
def get_FAN(num_HG:int, HG_depth:int, num_feats:int, resBlock:str, use_SE:bool):
    if resBlock == "HPM_ConvBlock":
        resBlock = HPM_ConvBlock
    elif resBlock == "Bottleneck":
        resBlock = Bottleneck
    elif resBlock == "InvertedResidual":
        resBlock = InvertedResidual

    return FAN(num_HG, HG_depth, num_feats, resBlock=resBlock, use_SE=use_SE)
    
def get_model(cfg:dict):
    model_type = cfg['model_type'][cfg['model_type_idx']]
    if model_type == "classifier":
        num_HG = cfg['num_HG']
        HG_depth = cfg['HG_depth']
        num_feats = cfg['num_feats']
        resBlock = cfg['resBlocks'][cfg['resBlock_idx']]
        use_SE = cfg['use_SE']
        return get_FAN(num_HG, HG_depth, num_feats, resBlock, use_SE)
    elif model_type == "regressor":
        backbone = cfg['backbone'][cfg['backbone_idx']]
        dropout = cfg['dropout']
        return RegressionModel(backbone, dropout=dropout)