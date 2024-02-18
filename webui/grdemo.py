import gradio as gr
from mmengine.logging import MMLogger
from utils import infer
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


image = gr.Image(label="Image", type="filepath", interactive=True) 
output = gr.Image(label='Result', interactive=False, elem_classes='result')




if __name__ == '__main__':
    title = 'Cell Mechanics Lab'

    DESCRIPTION = '''# <div align="center">High-throughput Drug Screen System</div>
    <div align="center">
    <img src="https://user-images.githubusercontent.com/45811724/190993591-
    bd3f1f11-1c30-4b93-b5f4-05c9ff64ff7f.gif" width="50%"/>
    </div>

    '''
    demo = gr.Interface(
    fn = infer,
    inputs = ["text", "text", image],
    outputs = output,
    title=title
)
    gr.Markdown(DESCRIPTION)
    demo.launch()
#     with gr.Blocks(analytics_enabled=False, title=title) as demo:
#         gr.Markdown(DESCRIPTION)
#         with gr.Tabs():
#             with gr.TabItem('Col Segmentation'):
#                 ColSegTab()
    


#     demo.launch(share=True)