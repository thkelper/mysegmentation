import gradio as gr
from mmengine.logging import MMLogger
# from utils import infer
import torch
from mmseg.apis import inference_model, init_model
import numpy as np
from mmengine.model import revert_sync_batchnorm
import cv2
import os
import os.path as osp
import time
from tools import save_txt_with_medicines, rm_not_valid_imgs_plot
from tools import get_fpath_list
from mmengine import Config
import pathlib

logger = MMLogger('mmdetection', logger_name='mmdet')
if torch.cuda.is_available():
    gpus = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    logger.info(f'Available GPUs: {len(gpus)}')
else:
    gpus = None
    logger.info('No available GPU.')


def get_free_device():
    if gpus is None:
        return torch.device('cpu')
    if hasattr(torch.cuda, 'mem_get_info'):
        free = [torch.cuda.mem_get_info(gpu)[0] for gpu in gpus]
        select = max(zip(free, range(len(free))))[1]
    else:
        import random
        select = random.randint(0, len(gpus) - 1)
    return gpus[select]


COL_MODEL_LIST = {
    "hrnet":['/home/yangchangpeng/wing_studio/mysegmentation/results/hrnet/fcn_hr18_4xb2-40k_cityscapes-512x1024.py', '/home/yangchangpeng/wing_studio/mysegmentation/results/hrnet/best_mIoU_iter_5104.pth']
}

AGA_MODEL_LIST = {
    "psanet":['/home/yangchangpeng/wing_studio/mysegmentation/results/single_cell/psanet/psanet_r50-d8_4xb4-20k_collagen-512x512.py', '/home/yangchangpeng/wing_studio/mysegmentation/results/single_cell/psanet/best_mIoU_iter_1276.pth']
}


class CollagenSeg:
    model_list = ["hrnet"]

    def __init__(self) -> None:
        self.create_ui()

    def create_ui(self):
        with gr.Column():
            with gr.Row():
                select_model = gr.Dropdown(
                    label="choose a model",
                    elem_id='od_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value=self.model_list[0]
                )

            with gr.Row():
                image_input = gr.Image(label="Image", 
                                 type="filepath", 
                                 interactive=True,
                                 container=False,
                                 sources="upload") 
                output = gr.Image(label='Result', 
                                  interactive=False)
            with gr.Row():
                run_button = gr.Button(
                        'RUN',
                        elem_classes='run_button'
                    )
                run_button.click(
                        self.infer,
                        inputs=[select_model, image_input],
                        outputs=output
                    )
                
    def infer(self, model, img):
        cfg_path = COL_MODEL_LIST[model][0]
        ckpt_path = COL_MODEL_LIST[model][1]
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
    
class AgaroseSeg:
    model_list = ["psanet"]

    def __init__(self) -> None:
        self.create_ui()

    def create_ui(self):
        with gr.Column():
            with gr.Row():
                select_model = gr.Dropdown(
                    label="choose a model",
                    elem_id='od_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value=self.model_list[0]
                )

            with gr.Row():
                image_input = gr.Image(label="Image", 
                                 type="filepath", 
                                 interactive=True,
                                 container=False,
                                 sources="upload") 
                output = gr.Image(label='Result', 
                                  interactive=False)
            with gr.Row():
                run_button = gr.Button(
                        'RUN',
                        elem_classes='run_button'
                    )
                run_button.click(
                        self.infer,
                        inputs=[select_model, image_input],
                        outputs=output
                    )
                
    def infer(self, model, img):
        cfg_path = AGA_MODEL_LIST[model][0]
        ckpt_path = AGA_MODEL_LIST[model][1]
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
    
class EC50Analysis:

    def __init__(self) -> None:
        self.create_ui()

    def create_ui(self):
        with gr.Column():
            with gr.Row():
                run_button = gr.Button('开始推理', elem_classes='run_button')
            with gr.Row():
                output = gr.Image(label='Result', interactive=False, type='filepath')
            run_button.click(
                        self.main,
                        inputs=[gr.Textbox(placeholder='config_path'), gr.Textbox(placeholder='Device')],
                        outputs=output
                    )
            
                
    def infer(self, cfgs, progress=gr.Progress()):
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

    def main(self, cfg_path, device=0):
        cfgs = Config.fromfile(cfg_path)
        cfgs.device = torch.device(f"cuda:{device}") if torch.cuda.is_available else 'cpu'
        os.makedirs(cfgs.infer_dir, exist_ok=True)


        if cfgs.infer:
            self.infer(cfgs)

        if cfgs.start_idx <= 1 and cfgs.end_idx >= 1:
            save_txt_with_medicines(cfgs, cfgs.infer_dir, cfgs.medicines)

        if cfgs.start_idx <= 2 and cfgs.end_idx >= 2:
            analysis_result_path = rm_not_valid_imgs_plot(cfgs, cfgs.infer_dir, all_medicines=cfgs.medicines, valid_thresh=cfgs.valid_thresh, verbose=cfgs.verbose)

            return pathlib.Path(analysis_result_path)

if __name__ == '__main__':
    title = 'Cell Mechanics Lab'

    DESCRIPTION = '''# <div align="center">Cell Mechanics Lab —— High-throughput Drug Screen System</div>
    <div align="center">
    <img src="https://kelper.cn/assets/img/projects/thesis/thesis_schema.png" width="50%"/>
    </div>

    '''

    with gr.Blocks(analytics_enabled=False, title=title, css="footer {visibility: hidden}") as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs():
            with gr.TabItem('Collagen Segmentation'):
                CollagenSeg()
            with gr.TabItem('Agarose Segmentation'):
                AgaroseSeg()
            with gr.TabItem('Batch Analysis'):
                EC50Analysis()

    demo.queue().launch()