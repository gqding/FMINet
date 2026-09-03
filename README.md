# FMINet

Official repository for **FMINet: Frequency Decoupled Cross-Modal Interaction Network for RGB-D Salient Object Detection**.

## Requirements

The code is implemented with PyTorch.

Recommended environment:

```bash
Python >= 3.8
PyTorch >= 1.12
torchvision
numpy
opencv-python
Pillow
tqdm
scipy
```

## Datasets

### RGB-D SOD

The experiments use the following RGB-D salient object detection datasets:

- NJU2K
- NLPR
- STERE
- SIP
- LFSD
- SSD

The suggested dataset organization is:

```text
datasets/
├── NJU2K/
│   ├── RGB/
│   ├── depth/
│   └── GT/
├── NLPR/
│   ├── RGB/
│   ├── depth/
│   └── GT/
├── STERE/
├── SIP/
├── LFSD/
└── SSD/
```

Please modify the dataset paths in the configuration files according to your local environment.

## How to run

Before training, download the required pretrained backbone weights and place them in the corresponding directory.

Example:

```bash
python main.py
```

Please check the script and configuration file for dataset paths, batch size, learning rate, number of epochs, and pretrained model paths.



## Pretrained Models

Download the trained FMINet checkpoints.

```text
checkpoints/
└── FMINet_best.pth
```

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{ding2026fminet,
  title={FMINet: Frequency Decoupled Cross-Modal Interaction Network for RGB-D Salient Object Detection},
  author={Ding, Guanqun and Chen, Ziyi and Liu, Weide and Fang, Yuming and Huang, Zizhi},
  year={2026}
}
```
