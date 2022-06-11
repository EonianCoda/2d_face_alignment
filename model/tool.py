from model.FAN import FAN
from model.FAN_boundary import Boundary_FAN
from model.FAN_SD import FAN_SD
from model.FAN_WSGN import FAN_WSGN
from model.FAN_aux import FAN_aux
from model.blocks import HPM_ConvBlock, Bottleneck, HPM_ConvBlock_WSGN, InvertedResidual
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
    output_CoordConv = cfg['output_CoordConv']
    # Boundary net
    add_boundary = cfg['add_boundary']
    # add stochastic dropout
    SD = cfg['SD']

    # group normalization
    use_gn = cfg['use_gn']
    use_ws = cfg['use_ws']

    aux_net = cfg['Aux_net']
    if aux_net:
        return FAN_aux(num_HG, HG_depth, num_feats, resBlock=resBlock, attention_block=attention_block,
                use_CoordConv=use_CoordConv, add_CoordConv_inHG=add_CoordConv_inHG, with_r=with_r)
    elif SD:
        return FAN_SD(num_HG, HG_depth, num_feats, attention_block=attention_block,use_CoordConv=use_CoordConv,
                     add_CoordConv_inHG=add_CoordConv_inHG, with_r=with_r)
    elif add_boundary:
        return Boundary_FAN(num_HG, HG_depth, num_feats, resBlock=resBlock, attention_block=attention_block,
                    with_r=with_r)
    elif use_gn or use_ws:
        return FAN_WSGN(num_HG, HG_depth, num_feats, resBlock=HPM_ConvBlock_WSGN, attention_block=attention_block, use_CoordConv=use_CoordConv, 
                        use_gn=use_gn, use_ws=use_ws, add_CoordConv_inHG=add_CoordConv_inHG, with_r=with_r)
    else:
        return FAN(num_HG, HG_depth, num_feats, resBlock=resBlock, attention_block=attention_block,
                use_CoordConv=use_CoordConv, add_CoordConv_inHG=add_CoordConv_inHG, with_r=with_r, 
                output_CoordConv=output_CoordConv)
