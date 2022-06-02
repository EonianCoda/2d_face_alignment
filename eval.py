import torch
from torch.utils.data import DataLoader
import argparse
from utils.evaluation import *
from model.tool import get_model
from utils.dataset import get_test_dataset
from utils.tool import load_parameters, val
from utils.visualize import plot_loss_68
from cfg import *
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_HG', type=int, default=4)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--type', type=str, default="val")
    args = parser.parse_args()

    fix_coord = cfg['fix_coord']
    ### path ###
    annot_path = f"./data/{args.type}_annot.pkl"
    data_path = f"./data/{args.type}"
    model_path = args.model_path
    ### model setting ###
    model_type = cfg['model_type'][cfg['model_type_idx']]
    if model_type == "classifier":
        cfg.update(classifier_cfg)
    elif model_type == "regressor":
        cfg.update(regressor_cfg)


    batch_size = cfg['batch_size'] * 2

    test_set = get_test_dataset(data_path, annot_path, model_type)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers= 2, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = get_model(cfg)

    load_parameters(model, model_path)
    test_NME_loss, test_NME_loss_68 = val(model, test_loader, device, model_type, fix_coord=fix_coord)
    print(f"Average NME Loss : {test_NME_loss:.4f}")
    plot_loss_68(test_NME_loss_68)
    print(np.argsort(test_NME_loss_68))

if __name__ == "__main__":
    main()