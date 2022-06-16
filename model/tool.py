from model.FAN import FAN
from model.blocks import CA_Block, SELayer

def get_model(cfg:dict):

    num_HG = cfg['num_HG']
    HG_depth = cfg['HG_depth']
    num_feats = cfg['num_feats']
    attention_block = cfg['attention_blocks'][cfg['attention_block_idx']]
    
    # Attention Block
    if attention_block == "None":
        attention_block = None
    elif attention_block == "SELayer":
        attention_block = SELayer
    elif attention_block == "CA_Block":
        attention_block = CA_Block
    use_CoordConv = cfg['use_CoordConv']
    add_CoordConv_inHG = cfg['add_CoordConv_inHG']
    with_r = cfg['with_r']

    return FAN(num_HG, HG_depth, num_feats, attention_block=attention_block,
            use_CoordConv=use_CoordConv, add_CoordConv_inHG=add_CoordConv_inHG, with_r=with_r)
