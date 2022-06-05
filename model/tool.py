from model.Regression import RegressionModel
from model.FAN import FAN
from model.blocks import HPM_ConvBlock, Bottleneck, InvertedResidual
from model.blocks import CA_Block, SELayer
def get_FAN(num_HG:int, HG_depth:int, num_feats:int, resBlock:str, attention_block:str):
    # Resdiual Block
    if resBlock == "HPM_ConvBlock":
        resBlock = HPM_ConvBlock
    elif resBlock == "Bottleneck":
        resBlock = Bottleneck
    elif resBlock == "InvertedResidual":
        resBlock = InvertedResidual
    # Attention Block
    if attention_block == "None":
        attention_block = None
    elif attention_block == "SELayer":
        attention_block = SELayer
    elif attention_block == "CA_Block":
        attention_block = CA_Block
    attention_block

    return FAN(num_HG, HG_depth, num_feats, resBlock=resBlock, attention_block=attention_block)
    
def get_model(cfg:dict):
    model_type = cfg['model_type'][cfg['model_type_idx']]
    if model_type == "classifier":
        num_HG = cfg['num_HG']
        HG_depth = cfg['HG_depth']
        num_feats = cfg['num_feats']
        resBlock = cfg['resBlocks'][cfg['resBlock_idx']]
        attention_block = cfg['attention_blocks'][cfg['attention_block_idx']]

        return get_FAN(num_HG, HG_depth, num_feats, resBlock, attention_block)
    elif model_type == "regressor":
        backbone = cfg['backbone'][cfg['backbone_idx']]
        dropout = cfg['dropout']
        return RegressionModel(backbone, dropout=dropout)