import torch
from torch.utils.data import DataLoader
import argparse
from utils.evaluation import *
from model.FAN import FAN
from model.Regression import RegressionModel
from utils.dataset import get_test_dataset
from utils.tool import load_parameters, val
from cfg import *


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_HG', type=int, default=4)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--type', type=str, default="val")
    args = parser.parse_args()

    ### path ###
    annot_path = f"./data/{args.type}_annot.pkl"
    data_path = f"./data/{args.type}"
    model_path = args.model_path
    ### model setting ###
    model_type = cfg['model_type'][cfg['model_type_idx']]
    if model_type == "classifier":
        cfg.update(classifier_cfg)
        num_HG = cfg['num_HG']
    elif model_type == "regressor":
        cfg.update(regressor_cfg)
        backbone = cfg['backbone'][cfg['backbone_idx']]
        dropout = cfg['dropout']

    batch_size = cfg['batch_size'] * 2

    test_set = get_test_dataset(data_path, annot_path, model_type)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers= 2, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    if model_type == "classifier":
        model = FAN(num_HG=num_HG)
    elif model_type == "regressor":
        model = RegressionModel(backbone, dropout=dropout)

    load_parameters(model, model_path)
    val(model, test_loader, device, model_type)


if __name__ == "__main__":
    main()