import torch
import os
import argparse
from dataset.FaceSynthetics import Heatmap_converter
from utils.evaluation import *
from model.tool import get_model
from dataset.tool import get_test_dataset
from utils.tool import load_parameters
from utils.visualize import read_img, plot_keypoints, Heatmap_visualizer
import matplotlib.pyplot as plt
import random
from cfg import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--plot_img', type=int, default=10)
    parser.add_argument('--annot_path', type=str, default="./data/val_annot.pkl")
    parser.add_argument('--data_path', type=str, default="./data/val")
    parser.add_argument('--show_index', action="store_false")
    parser.add_argument('--show_line', action="store_false")
    parser.add_argument('--show_bad', action="store_false")
    parser.add_argument('--bad_loss', type=float, default=2.0)
    args = parser.parse_args()

    
    ### Data parameters ##
    annot_path = args.annot_path
    data_path = args.data_path
    model_path = args.model_path
    ### image parameters ##
    show_line = args.show_line
    show_index = args.show_index
    show_bad = args.show_bad
    bad_loss = args.bad_loss
    ### model setting ###

    fix_coord = cfg['fix_coord']
    test_set = get_test_dataset(data_path, annot_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create model
    model = get_model(cfg)

    load_parameters(model, model_path)

    heatmap_visualizer = Heatmap_visualizer()
    heatmap_converter = Heatmap_converter(96)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        idxs = [i for i in range(len(test_set))]
        random.shuffle(idxs)
        #idxs = idxs[: args.plot_img]

        num_cur_show = 0
        for i in idxs:
            img_path = test_set.images[i]
            
            sample = test_set.__getitem__(i)
            img, gt_label = sample['img'], sample['gt_label']
            img = img.to(device).unsqueeze(dim=0)
            outputs = model(img)
            pred = heatmap_to_landmark(outputs, fix_coord=fix_coord)
          
            pred = pred[0]
            NME_loss = NME(pred, gt_label)
            if (show_bad and NME_loss >= bad_loss) or not show_bad:
                # Draw keypoints on the image
                origin_im = read_img(os.path.join(data_path, img_path))
                # Points
                fig, axs = plt.subplots(1,3,figsize=(18,6))
                im = plot_keypoints(origin_im.copy(), gt_label, pred, show_index, show_line)
                axs[0].imshow(im)
                axs[0].set_title(f"Loss = {NME_loss:4f}" + img_path)
                # Predicting Heatmap
                heatmap_visualizer.draw_heatmap(origin_im.copy(), outputs[-1], color="red", ax=axs[1])
                axs[1].set_title("Pred Heatmap")
                # Groud Truth heatmap
                heatmap_visualizer.draw_heatmap(origin_im.copy(), heatmap_converter.convert(gt_label), color="red", ax=axs[2])
                axs[2].set_title("Ground Truth Heatmap")
                num_cur_show += 1
                if num_cur_show == args.plot_img:
                    break
    plt.show()

if __name__ == "__main__":
    main()