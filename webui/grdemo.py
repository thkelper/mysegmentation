import gradio as gr
from mmengine.logging import MMLogger
from .utils import infer
import torch

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


class ColSegTab:
    model_list = [
        'retinanet_r50-caffe_fpn_1x_coco',
        'faster-rcnn_r50-caffe_fpn_1x_coco',
        'dino-5scale_swin-l_8xb2-12e_coco.py',
    ]

    def __init__(self) -> None:
        self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='od_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value=self.model_list[0],
                )
            with gr.Column():
                image_input = gr.Image(
                    label='Image',
                    source='upload',
                    elem_classes='input_image',
                    type='filepath',
                    interactive=True,
                    tool='editor',
                )
                output = gr.Image(
                    label='Result',
                    source='upload',
                    interactive=False,
                    elem_classes='result',
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, image_input],
                    outputs=output,
                )


    def inference(self, model, image):
        det_inferencer = infer(
            model, scope='mmdet', device=get_free_device())
        results_dict = det_inferencer(image, return_vis=True, no_save_vis=True)
        vis = results_dict['visualization'][0]
        return vis


if __name__ == '__main__':
    title = 'Cell Mechanics Lab'

    DESCRIPTION = '''# <div align="center">MMDetection Inference Demo  </div>
    <div align="center">
    <img src="https://user-images.githubusercontent.com/45811724/190993591-
    bd3f1f11-1c30-4b93-b5f4-05c9ff64ff7f.gif" width="50%"/>
    </div>

    #### This is an official demo for MMDet. \n

    - The first time running requires downloading the weights,
    please wait a moment. \n
    - OV is mean Open Vocabulary \n
    - Refer Seg is mean Referring Expression Segmentation \n
    - In Text-Image Region Retrieval, you need to provide n images and
    a query text, and the model will predict the most matching image and
    its corresponding grounding mask.
    '''

    with gr.Blocks(analytics_enabled=False, title=title) as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs():
            with gr.TabItem('Detection'):
                ColSegTab()


    demo.queue().launch(share=True)