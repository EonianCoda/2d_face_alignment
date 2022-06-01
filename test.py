
import torch
from torch.utils.data import DataLoader
from utils.dataset import get_pred_dataset
from utils.tool import load_parameters
from utils.visualize import read_img, plot_keypoints
from utils.evaluation import *
from model.FAN import FAN
from model.Regression import RegressionModel
from cfg import *
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import os
import random

def shwo_img(img_path, label):
    im = read_img(img_path)
    im = plot_keypoints(im, pred=label, show_line=False)
    plt.figure()
    plt.imshow(im)

def pred_imgs(model, test_loader, model_type:str, device):
    model = model.to(device)
    preds = []

    for batch_idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            data = data.to(device)
            outputs = model(data)

            if model_type == "classifier":
                pred = heatmap_to_landmark(outputs)
            elif model_type == "regressor":
                pred = outputs.detach().cpu()
            
            preds.append(pred)
    preds = torch.cat(preds, dim=0)
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--type', type=str, default="val")
    args = parser.parse_args()

    ### path ###
    data_path = f"./data/{args.type}"
    model_path = args.model_path
    ### model setting ###
    model_type = cfg['model_type'][cfg['model_type_idx']]
    if model_type == "classifier":
        cfg.update(classifier_cfg)
        num_HG = cfg['num_HG']
        HG_depth = cfg['HG_depth']
    elif model_type == "regressor":
        cfg.update(regressor_cfg)
        backbone = cfg['backbone'][cfg['backbone_idx']]
        dropout = cfg['dropout']

    batch_size = cfg['batch_size'] * 2
    test_set = get_pred_dataset(data_path)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers= 2, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create model
    if model_type == "classifier":
        model = FAN(num_HG=num_HG,HG_depth = HG_depth)
    elif model_type == "regressor":
        model = RegressionModel(backbone, dropout=dropout)

    load_parameters(model, model_path)
    preds = pred_imgs(model=model, 
                test_loader=test_loader, 
                device=device,
                model_type=model_type)
    images = test_set.images

    # Visualize some image for checking
    idxs = [i for i in range(len(images))]
    random.shuffle(idxs)
    for i in range(10):
        i = idxs[i]
        shwo_img(os.path.join(data_path, images[i]), pred[i])
        




    lines = []
    formated_str = "{:.4f} {:.4f}"
    for i, (img_name, pred) in enumerate(zip(images, preds)):
        
        output = [img_name]
        for (x, y) in pred:
            output.append(formated_str.format(x, y))
        line = " ".join(output)
        if i != len(images):
            line += '\n'
        lines.append(line)
    with open("solution.txt", 'w') as f:
        f.writelines(lines)
if __name__ == "__main__":
    main()