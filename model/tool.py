from model.FAN import FAN
from model.blocks import HPM_ConvBlock, Bottleneck, InvertedResidual
from model.blocks import CA_Block, SELayer

def get_model(cfg:dict):

    num_HG = cfg['num_HG']
    HG_depth = cfg['HG_depth']
    num_feats = cfg['num_feats']
    resBlock = cfg['resBlocks'][cfg['resBlock_idx']]
    attention_block = cfg['attention_blocks'][cfg['attention_block_idx']]

    # Residual Block
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
    use_CoordConv = cfg['use_CoordConv']
    add_CoordConv_inHG = cfg['add_CoordConv_inHG']
    with_r = cfg['with_r']
    return FAN(num_HG, HG_depth, num_feats, resBlock=resBlock, attention_block=attention_block,
            use_CoordConv=use_CoordConv, add_CoordConv_inHG=add_CoordConv_inHG, with_r=with_r)
