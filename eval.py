import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import argparse
from utils.evaluation import *
from utils.model import FAN
from utils.dataset import get_transform, process_annot, FaceSynthetics
from utils.tool import load_parameters
from cfg import cfg
from tqdm import tqdm
def val(model, test_set, batch_size:int, device):
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers= 2, pin_memory=True)
    
    total_pred_loss = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, label, gt_label) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            data = data.to(device)
            label = label.to(device)
            outputs = model(data)
            loss = 0
            for output in outputs:
                loss += criterion(output, label)
            pred = heatmap_to_landmark(outputs)
            pred_loss = NME(pred, gt_label)
            total_pred_loss += pred_loss
    print(total_pred_loss / len(test_loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_HG', type=int, default=4)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--annot_path', type=str, default="./data/val_annot.pkl")
    parser.add_argument('--data_path', type=str, default="./data/val")
    args = parser.parse_args()

    annot_path = args.annot_path
    data_path = args.data_path
    batch_size = cfg['batch_size']

    images, labels, gt_labels = process_annot(annot_path)
    test_set = FaceSynthetics(data_path, images, labels, gt_labels, get_transform("test"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FAN(num_HG=args.num_HG)
    load_parameters(model, args.model_path)
    model = model.to(device)
    model.eval()
    val(model, test_set, batch_size, device)


if __name__ == "__main__":
    main()