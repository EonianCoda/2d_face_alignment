import torch
from torch.utils.data import DataLoader
import argparse
from utils.evaluation import *
from model.FAN import FAN
from model.Regression import RegressionModel
from utils.dataset import get_test_dataset
from utils.tool import load_parameters, val
from cfg import cfg


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
    model_type = cfg['model_type']
    backbone = cfg['backbone']
    num_HG = cfg['num_HG']

    batch_size = cfg['batch_size']


    test_set = get_test_dataset(data_path, annot_path, model_type)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers= 2, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    if model_type == "classifier":
        model = FAN(num_HG=num_HG)
    elif model_type == "regressor":
        model = RegressionModel(backbone)

    load_parameters(model, model_path)
    val(model, test_loader, device, model_type)


if __name__ == "__main__":
    main()