import torch
import os
import argparse
from utils.evaluation import *
from model.tool import get_model
from utils.dataset import get_test_dataset
from utils.tool import load_parameters
from utils.visualize import read_img, plot_keypoints
import matplotlib.pyplot as plt
import random
from cfg import *


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_HG', type=int, default=4)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--type', type=str, default="val")
    parser.add_argument('--plot_img', type=int, default=10)
    parser.add_argument('--show_index', action="store_false")
    parser.add_argument('--show_line', action="store_true")
    args = parser.parse_args()

    
    ### Data parameters ##
    annot_path = f"./data/{args.type}_annot.pkl"
    data_path = f"./data/{args.type}"
    model_path = args.model_path
    ### image parameters ##
    show_line = args.show_line
    show_index = args.show_index

    ### model setting ###
    model_type = cfg['model_type'][cfg['model_type_idx']]
    if model_type == "classifier":
        cfg.update(classifier_cfg)
    elif model_type == "regressor":
        cfg.update(regressor_cfg)

    fix_coord = cfg['fix_coord']
    test_set = get_test_dataset(data_path, annot_path, model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create model
    model = get_model(cfg)

    load_parameters(model, model_path)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        idxs = [i for i in range(len(test_set))]
        random.shuffle(idxs)
        idxs = idxs[: args.plot_img]
        for i in idxs:
            img_path = test_set.images[i]
            img, _ , gt_label = test_set.__getitem__(i)
            img = img.to(device).unsqueeze(dim=0)
            outputs = model(img)
            if model_type == "classifier":
                pred = heatmap_to_landmark(outputs, fix_coord=fix_coord)
            elif model_type == "regressor":
                pred = outputs.detach().cpu()
            pred = pred[0]
            NME_loss = NME(pred, gt_label)

            # Draw keypoints on the image
            im = read_img(os.path.join(data_path, img_path))
            im = plot_keypoints(im, gt_label, pred, show_index, show_line)
            plt.figure()
            plt.title(f"Loss = {NME_loss:4f}")
            plt.imshow(im)
            if i == args.plot_img:
                break
    plt.show()

if __name__ == "__main__":
    main()