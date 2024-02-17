import os
import os.path as osp
from mmengine import Config
import torch
from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
import numpy as np
import cv2
from tools.utils import get_fpath_list

IMG_EXT = ".bmp"
MASK_EXT = ".png"
INFER_EXT = ".png"

def infer(cfg_path, ckpt_path, img):
    # build the model from a config file and a checkpoint file
    if torch.cuda.is_available():
        device = torch.device(f'cuda:0')
    else:
        device = 'cpu'
    model = init_model(cfg_path, ckpt_path, device=device)
    if device == 'cpu':
        model = revert_sync_batchnorm(model)

    result = inference_model(model, img)
    
    mask = result.pred_sem_seg.cpu().data.numpy()
    mask = np.transpose(np.repeat(mask, 3, axis=0), (1, 2, 0))
    # mask[mask==1] = 255
    
    return mask