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

## Training

Before training, download the required pretrained backbone weights and place them in the corresponding directory.

Example:

```bash
python train.py
```

Please check the training script and configuration file for dataset paths, batch size, learning rate, number of epochs, and pretrained model paths.

## Testing

Run the inference script with the trained checkpoint:

```bash
python test.py
```

The predicted saliency maps will be saved to the configured output directory.


## Pretrained Models

The trained FMINet checkpoints can be downloaded from the link.

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
