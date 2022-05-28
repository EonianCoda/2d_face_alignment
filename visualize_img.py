import torch
import os
import argparse
from utils.evaluation import *
from utils.model import FAN
from utils.dataset import process_annot, FaceSynthetics
from utils.tool import load_parameters
from utils.visualize import read_img, plot_keypoints
import matplotlib.pyplot as plt
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_HG', type=int, default=4)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--type', type=str, default="val")
    parser.add_argument('--plot_img', type=int, default=10)
    parser.add_argument('--show_index', action="store_false")
    parser.add_argument('--show_line', action="store_false")
    args = parser.parse_args()

    # Data parameters
    annot_path = f"./data/{args.type}_annot.pkl"
    data_path = f"./data/{args.type}"
    # image parameters
    show_line = args.show_line
    show_index = args.show_index

    images, labels, gt_labels = process_annot(annot_path)
    test_set = FaceSynthetics(data_path, images, labels, gt_labels, "test")

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