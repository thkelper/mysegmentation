# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import cv2
import numpy as np
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot
from utils import get_fpath_list

def parse_args():
    parser = ArgumentParser(description="批量图像推理")
    parser.add_argument('-s', '--src_img_dir', required=True, help='Image files directory')
    parser.add_argument('-cfg', '--config', required=True, help='Config file')
    parser.add_argument('-ckpt', '--checkpoint', required=True, help='Checkpoint file')
    parser.add_argument('-t', '--tar_dir', required=True, default=None, help='Directory to output file')
    parser.add_argument(
        '--img_ext', default='.bmp', help='the ext of images')
    parser.add_argument(
        '--mask_ext', default='.png', help='the ext of masks')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()
    return args

def infer(args):
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    img_paths = get_fpath_list(args.src_img_dir)
    for img_path in img_paths:
        result = inference_model(model, img_path)
        img_fname = osp.basename(img_path)
        img_title = osp.splitext(img_fname)[0]
        mask_fname = img_fname.replace(args.img_ext, args.mask_ext)
        out_path = osp.join(args.tar_dir, mask_fname)
        mask = result.pred_sem_seg.cpu().data.numpy()
        mask = np.transpose(np.repeat(mask, 3, axis=0), (1, 2, 0))
        mask[mask==1] = 255
        # import pdb
        # pdb.set_trace()
        cv2.imwrite(out_path, mask)
        
    # show the results
        # mask = show_result_pyplot(
        #     model,
        #     img_path,
        #     result,
        #     title=img_title,
        #     opacity=args.opacity,
        #     with_labels=args.with_labels,
        #     draw_gt=False,
        #     show=False if out_path is not None else True,
        #     out_file=out_path)
def main():
    args = parse_args()
    infer(args)

if __name__ == '__main__':
    infer()
