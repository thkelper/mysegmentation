import os.path as osp

infer = True

data_root_dir = "/home/yangchangpeng/wing_studio/data/col_dataset"
img_dir = osp.join(data_root_dir, "img_dir")
infer_dir = osp.join(data_root_dir, "infer_dir")

model_cfg_path = "/home/yangchangpeng/wing_studio/mysegmentation/results/hrnet/fcn_hr18_4xb2-40k_cityscapes-512x1024.py"
model_ckpt_path = "/home/yangchangpeng/wing_studio/mysegmentation/results/hrnet/best_mIoU_iter_5104.pth"
IMG_EXT = ".bmp"
MASK_EXT = ".png"
INFER_EXT = ".png"
# medicines = ['1uM_Ach', '1uM_Ach_0.0001nM_TB', '1uM_Ach_0.001nM_TB', '1uM_Ach_0.01nM_TB', '1uM_Ach_0.1nM_TB',
            #  '1uM_Ach_1nM_TB', '1uM_Ach_10nM_TB', '1uM_Ach_100nM_TB', '1uM_Ach_1000nM_TB', 'Control']
# medicines = ['1uM_Ach', '1uM_Ach_0.0001nM_TB', '1uM_Ach_0.001nM_TB', '1uM_Ach_0.01nM_TB', '1uM_Ach_0.1nM_TB',
#              '1uM_Ach_1nM_TB', '1uM_Ach_10nM_TB', '1uM_Ach_100nM_TB', '1uM_Ach_1000nM_TB'] 
medicines = None

valid_thresh = 2
start_idx = 0 

# infer = True
end_idx = 0

# infer_configs 
infer_ann_file = osp.join(infer_dir, "ann_info.txt")
vis_feature = False
verbose = True