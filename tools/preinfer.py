from argparse import ArgumentParser
import os
import os.path as osp
from wzi_col_utils import (save_txt_with_medicines, rm_not_valid_imgs_plot)
from mmengine import Config

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model
import numpy as np
import cv2
from utils import get_fpath_list

def infer(cfgs):
    # build the model from a config file and a checkpoint file
    model = init_model(cfgs.model_cfg_path, cfgs.model_ckpt_path, device=cfgs.device)
    if cfgs.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    img_paths = get_fpath_list(cfgs.img_dir, valid_ext=cfgs.IMG_EXT, recursive=True)
    for img_path in img_paths:
        result = inference_model(model, img_path)
        img_fname = osp.basename(img_path)
        mask_fname = img_fname.replace(cfgs.IMG_EXT, cfgs.MASK_EXT)
        # import pdb
        # pdb.set_trace()
        # out_path = osp.join(cfgs.infer_dir, mask_fname)
        mask_path = img_path.replace(cfgs.img_dir, cfgs.infer_dir)
        mask_dir = osp.dirname(mask_path)
        os.makedirs(mask_dir, exist_ok=True)    
        mask = result.pred_sem_seg.cpu().data.numpy()
        mask = np.transpose(np.repeat(mask, 3, axis=0), (1, 2, 0))
        mask[mask==1] = 255
        # import pdb
        # pdb.set_trace()
        cv2.imwrite(mask_path, mask)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_ann_txt(img_dir, ann_save_dir, img_ext=".bmp", mask_ext=".tif", mode="train"):
    all_ann_info = []
    for cur_root, dirs, files in os.walk(img_dir):
        for file in files:
            fpath = osp.join(cur_root, file)
            if img_ext in fpath and not "._" in fpath:
                mask_path = fpath.replace(img_ext, mask_ext)
                if mode=="train" and osp.exists(mask_path):
                    all_ann_info.append(fpath + " " + mask_path + "\n")
                elif mode=="infer":
                    # all_ann_info.append(fpath + " " + mask_path + "\n")
                    file_relative_path = osp.relpath(fpath, start=img_dir)
                    all_ann_info.append(file_relative_path + "\n") 
                else:
                    print(mask_path)
                    continue
    if mode == "train":
        with open(osp.join(ann_save_dir, "ann_info.txt"), "w") as f:
            f.writelines(all_ann_info)
    if mode == "infer":
        with open(osp.join(ann_save_dir, "infer_info.txt"), "w") as f:
            f.writelines(all_ann_info)

def read_txt(fpath):
    img_list = list()
    with open(fpath, "r") as f:
        lines = f.readlines()
        for curline in lines:
            curline = curline.strip()
            img_list.append(curline)
    return img_list


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config_path', type=str, default="./preinfer_configs.py")
#     return parser.parse_args()


def parse_args():
    parser = ArgumentParser(description="胶原凝胶或者琼脂糖图像推理分析")
    parser.add_argument('--config', type=str, help="所有通用配置文件" )
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfgs = Config.fromfile(args.config)
    cfgs.device = args.device
    os.makedirs(cfgs.infer_dir, exist_ok=True)
    # 产生工作目录下所有文件的txt
    # if cfgs.start_idx <= 0 and cfgs.end_idx >= 0:
    #     generate_ann_txt(cfgs.img_dir, cfgs.infer_dir, img_ext=cfgs.IMG_EXT, mode="infer")

    if cfgs.infer:
        infer(cfgs)

    if cfgs.start_idx <= 2 and cfgs.end_idx >= 2:
        save_txt_with_medicines(cfgs, cfgs.root_dir, cfgs.medicines)

    if cfgs.start_idx <= 3 and cfgs.end_idx >= 3:
        rm_not_valid_imgs_plot(cfgs, cfgs.root_dir, all_medicines=cfgs.medicines, valid_thresh=cfgs.valid_thresh, verbose=cfgs.verbose)

    

if __name__ == "__main__":
    # root_dir = "/mnt/disk1/data/mouse_det/month3"
    # root_dir = "/mnt/disk1/data/mouse_det/0329_11am_cell800k_col1.0"
    # root_dir = "/mnt/d/ycp/pku/unet++/input/anti_cancer_phase"
    # root_dir = "/mnt/d/ycp/pku/unet++/input/anti_cancer_phase/0628_48well"
    
    main()
    # ann_file_path = osp.join(root_dir, "ann_info.txt")
    # test_list = read_txt(ann_file_path)