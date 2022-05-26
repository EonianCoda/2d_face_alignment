import torch
import os
import argparse
from utils.evaluation import *
from utils.model import FAN
from utils.dataset import get_transform, process_annot, FaceSynthetics
from utils.tool import load_parameters, plot_keypoint
import matplotlib.pyplot as plt
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_HG', type=int, default=4)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--annot_path', type=str, default="./data/val_annot.pkl")
    parser.add_argument('--data_path', type=str, default="./data/val")
    parser.add_argument('--plot_img', type=int, default=10)
    args = parser.parse_args()

    annot_path = args.annot_path
    data_path = args.data_path


    images, labels, gt_labels = process_annot(annot_path)
    test_set = FaceSynthetics(data_path, images, labels, gt_labels, get_transform("test"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FAN(num_HG=args.num_HG)
    load_parameters(model, args.model_path)

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
            pred = heatmap_to_landmark(outputs)

            im = plot_keypoint(os.path.join(data_path, img_path), gt_label, pred[0].tolist())

            plt.figure()
            plt.imshow(im)
            if i == args.plot_img:
                break
    plt.show()

if __name__ == "__main__":
    main()