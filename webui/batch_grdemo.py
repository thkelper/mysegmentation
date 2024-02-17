import gradio as gr
from mmengine.logging import MMLogger
from utils import infer
import torch
from mmseg.apis import inference_model, init_model
import numpy as np
from mmengine.model import revert_sync_batchnorm

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
                                 container=False) 
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
                                 container=False) 
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

if __name__ == '__main__':
    title = 'Cell Mechanics Lab'

    DESCRIPTION = '''# <div align="center">Cell Mechanics Lab —— High-throughput Drug Screen System</div>
    <div align="center">
    <img src="https://kelper.cn/assets/img/projects/thesis/thesis_schema.png" width="50%"/>
    </div>

    '''

    with gr.Blocks(analytics_enabled=False, title=title) as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs():
            with gr.TabItem('Collagen Segmentation'):
                CollagenSeg()
            with gr.TabItem('Agarose Segmentation'):
                AgaroseSeg()

    demo.queue().launch()