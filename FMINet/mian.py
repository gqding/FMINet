from torch.utils import data
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2, torch
from tools.cfg import py2cfg
from tools.cltfun_tool import collate_fn
from rgbdsod.datasets.dataset_rgbd_sod import SalObjDataset
from train_supervision import Supervision_Train
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id',   type=str, default='0', help='select gpu id')
# parser.add_argument('--test_path',type=str, default='RGBD_dataset/TestDataset/',help='test dataset path')
parser.add_argument('--test_path',type=str, default='/gemini/data-1/RGBD_dataset/TestDataset/',help='test dataset path')
parser.add_argument("-c", "--config_path", default='config/config_cacnet.py',
    help="Path to the config.")
opt = parser.parse_args()
config = py2cfg(opt.config_path)
dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
 

#load the model
trained_model_path=os.path.join(config.weights_path, config.test_weights_name + '.ckpt')
model = Supervision_Train.load_from_checkpoint(trained_model_path, config=config)
print("Loaded trained {} model from: {}".format(config.test_weights_name, trained_model_path))

model.cuda()
model.eval()

#test
test_datasets = ['NJU2K','NLPR', 'DES', 'SSD','SIP', 'STERE',
                 'DUT-RGBD', 'LFSD', 'ReDWebTest']


for dataset in test_datasets:
    save_path = 'results/test_maps/' + config.test_weights_name + '/' + dataset + '/'
    # save_path = 'results/test_maps/' + config.test_weights_name + '/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    image_root  = dataset_path + dataset + '/RGB/'
    gt_root     = dataset_path + dataset + '/GT/'
    depth_root  = dataset_path + dataset + '/depth/'

    dataset = SalObjDataset(image_root, gt_root, depth_root, opt.testsize, mode='test')

    test_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,  # config.val_batch_size
                                  shuffle=True,
                                  num_workers=8, collate_fn=collate_fn,
                                  pin_memory=False,
                                  drop_last=False,
                                  )
    t_bar=tqdm(test_loader)
    for input in t_bar:

        image, gt,depth, names = input['img'], input['gt'], input['depth'], input["img_id"]

        raw_predictions_all = model(image.cuda(), depth.cuda())
        raw_predictions = raw_predictions_all

        prediction_map_logits, x_output_0, x_output_1, x_output_2 = raw_predictions

        pre_mask = prediction_map_logits.sigmoid()
        pre_mask = (pre_mask - pre_mask.min()) / (pre_mask.max() - pre_mask.min() + 1e-8)
        pre_boundary = x_output_0.sigmoid().data.cpu().numpy()

        for i in range(pre_mask.shape[0]):
            pred_seg = pre_mask[i].unsqueeze(0).float()  # [1, 1, h, w]
            raw_h, raw_w=input['info'][i]['raw_height'], input['info'][i]['raw_width']
            pred_seg=F.interpolate(pred_seg, size=[raw_h, raw_w], mode='bilinear', align_corners=False)

            pred_seg = pred_seg.squeeze().detach().cpu().numpy()

            masks_true_orig = F.interpolate(gt[i].unsqueeze(0).float(), size=[raw_h, raw_w], mode='bilinear', align_corners=False)

            save_name=os.path.join(save_path, names[i].replace('.jpg', '.png'))
            t_bar.set_description('save img to: '+save_name)
            cv2.imwrite(save_name,np.uint8(pred_seg*255))

print('Test Done!')
