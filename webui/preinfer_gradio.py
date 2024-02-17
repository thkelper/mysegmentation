from tools import save_txt_with_medicines, rm_not_valid_imgs_plot
from tools import get_fpath_list
import os
import os.path as osp
import cv2
import torch
from mmseg.apis import inference_model, init_model
import numpy as np
from mmengine.model import revert_sync_batchnorm
from mmengine import Config
import gradio as gr
import pathlib
import time

def infer(cfgs, progress=gr.Progress()):
    # build the model from a config file and a checkpoint file
    progress(0, desc="Starting...")
    time.sleep(1)
    model = init_model(cfgs.model_cfg_path, cfgs.model_ckpt_path, device=cfgs.device)
    if cfgs.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    img_paths = get_fpath_list(cfgs.img_dir, valid_ext=cfgs.IMG_EXT, recursive=True)
    for img_path in progress.tqdm(img_paths):
        result = inference_model(model, img_path)
        img_fname = osp.basename(img_path)
        mask_fname = img_fname.replace(cfgs.IMG_EXT, cfgs.MASK_EXT)
        # import pdb
        # pdb.set_trace()
        # out_path = osp.join(cfgs.infer_dir, mask_fname)
        mask_path = img_path.replace(cfgs.img_dir, cfgs.infer_dir)
        mask_dir = osp.dirname(mask_path)
        
        os.makedirs(mask_dir, exist_ok=True)    
        mask_path = mask_path.replace(cfgs.IMG_EXT, cfgs.INFER_EXT)
        mask = result.pred_sem_seg.cpu().data.numpy()
        mask = np.transpose(np.repeat(mask, 3, axis=0), (1, 2, 0))
        mask[mask==1] = 255
        # import pdb
        # pdb.set_trace()
        cv2.imwrite(mask_path, mask)


def main(cfg_path, device=0):
    cfgs = Config.fromfile(cfg_path)
    cfgs.device = torch.device(f"cuda:{device}") if torch.cuda.is_available else 'cpu'
    os.makedirs(cfgs.infer_dir, exist_ok=True)


    if cfgs.infer:
        infer(cfgs)

    if cfgs.start_idx <= 1 and cfgs.end_idx >= 1:
        save_txt_with_medicines(cfgs, cfgs.infer_dir, cfgs.medicines)

    if cfgs.start_idx <= 2 and cfgs.end_idx >= 2:
        analysis_result_path = rm_not_valid_imgs_plot(cfgs, cfgs.infer_dir, all_medicines=cfgs.medicines, valid_thresh=cfgs.valid_thresh, verbose=cfgs.verbose)

        return pathlib.Path(analysis_result_path)
        
output = gr.Image(label='Result', interactive=False, type='filepath')

demo = gr.Interface(
    fn=main,
    inputs=["text", "text"],
    outputs=output
)

demo.launch()



