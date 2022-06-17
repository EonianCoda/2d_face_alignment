# 2D landmark detection
**#2022 Computer Vision Final Project**<br>
**#Microsoft Light-Weight Facial Landmark Prediction Challenge**<br>
68-point facial landmarks prediction
## Environment
```
python >= 3.9.12
torch >= 1.11.0
torchvision >= 0.12.0
```
### Installation
```bash
pip3 install -r requirement
```
Pytorch 
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
## Usage
### Change Data Path
```
├── 2d_landmark_detection
│   ├── dataset
│   ├── model
│       ....
├── data
│   ├── aflw_val            #your train root path
|   |   ├── annot.pkl       #your train annotation path
|   |       ....
│   └── synthetics_train    #your test root path
|       ├── annot.pkl       #your test annotation path
|           ....
└── aflw_test
```
`cfg.py`
```PYTHON
    cfg={
        .
        .

        'train_annot':'your train annotation path',  # default = '../data/synthetics_train/annot.pkl'
        'train_data_root':'your train root path',    # default = '../data/synthetics_train'
        .
        .
        'val_annot':'your val annotation path',       # default = '../data/aflw_val/annot.pkl' 
        'val_data_root':'your val root path',         # default = '../data/aflw_val' 
        .
        ....
    }
```
### Trainging
Usuage：
```bash
python main.py <option>
```
#### Options
* `--use_image_ratio` how much image ration yot want to use（default = 1.0） 
* `--exp_name`  the name of experiment on Tensorboard
* `--resume` resume training model
* `--resume_epoch` the epoch you want to resume
* `--resume_model_epoch` the path of model weight for resuming training
### Testing
Usuage：
```bash
python test.py <option>
```
and you will get **solution.zip**
#### Option
* `--model_path` the path of model you want to predict（default = best.pt） 
* `--data_path` the path of test data （default = ../aflw_test） 
* `--show_result`  （default = False) 

### Visualize
Usuage：
```bash
python visualize_img.py <option>
```
#### Option
* `--model_path` the path of model you want to predict（default = best.pt） 
* `--plot_img` number of image you want to visualize
* `--annot_path` annotation of img you want to visualize（default = ../data/aflw_val/annot.pkl）
* `--data_path` the path of test data （default = ../data/aflw_val） 
* `--show_index` the index of landmark
* `--show_line` the distance of predict and ground truth
* `--show_bad` showing the image which loss greater than bad_loss
* `--bad_loss` the threshold of show_bad（default = 2.0） 

## Reference
1. A. Newell, K. Yang, and J. Deng. Stacked hourglass networks for human pose estimation. In ECCV, 2016.
2. Bulat, Adrian, and Georgios Tzimiropoulos. "How far are we from solving the 2d & 3d face alignment problem?(and a dataset of 230,000 3d facial landmarks)." Proceedings of the IEEE International Conference on Computer Vision. 2017.
3. Feng, Zhen-Hua, et al. "Wing loss for robust facial landmark localisation with convolutional neural networks."      Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
4. Wang, Xinyao, Liefeng Bo, and Li Fuxin. "Adaptive wing loss for robust face alignment via heatmap regression." Proceedings of the IEEE/CVF international conference on computer vision. 2019. 
5. Hu, J., Shen, L., and Sun, G. Squeeze-and-excitation networks. CVPR, 2018.
6. Qibin Hou, Daquan Zhou, and Jiashi Feng. Coordinate attention for efficient mobile network design. arXiv preprint arXiv:2103.02907, 2021.
7. Liu, Rosanne, et al. "An intriguing failing of convolutional neural networks and the coordconv solution." Advances in neural information processing systems 31 (2018).
8. Guo, Xiaojie, et al. "PFLD: A practical facial landmark detector." arXiv preprint arXiv:1902.10859 (2019).
9. Wu, Wayne, et al. "Look at boundary: A boundary-aware face alignment algorithm." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
