import os.path as osp

infer = True

# data_root_dir = "/home/yangchangpeng/wing_studio/data/col_dataset"
data_root_dir = "/home/yangchangpeng/wing_studio/data/"
img_dir = osp.join(data_root_dir, "20240201_ASMC_single_cell_test")
infer_dir = osp.join(data_root_dir, "infer_test_20240201_ASMC_single_cell")

model_cfg_path = "/home/yangchangpeng/wing_studio/mysegmentation/results/single_cell/lr_0.01/psanet_r50-d8_4xb4-20k_collagen-512x512.py"
model_ckpt_path = "/home/yangchangpeng/wing_studio/mysegmentation/results/single_cell/lr_0.01/best_mIoU_iter_5104.pth"
IMG_EXT = ".tif"
MASK_EXT = ".png"
INFER_EXT = ".png"
# medicines = ['1uM_Ach', '1uM_Ach_0.0001nM_TB', '1uM_Ach_0.001nM_TB', '1uM_Ach_0.01nM_TB', '1uM_Ach_0.1nM_TB',
            #  '1uM_Ach_1nM_TB', '1uM_Ach_10nM_TB', '1uM_Ach_100nM_TB', '1uM_Ach_1000nM_TB', 'Control']
# medicines = ['1uM_Ach', '1uM_Ach_0.0001nM_TB', '1uM_Ach_0.001nM_TB', '1uM_Ach_0.01nM_TB', '1uM_Ach_0.1nM_TB',
#              '1uM_Ach_1nM_TB', '1uM_Ach_10nM_TB', '1uM_Ach_100nM_TB', '1uM_Ach_1000nM_TB'] 
# medicines = None
medicines = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']

valid_thresh = 6
start_idx = 0 

# infer = True
end_idx = 0

# infer_configs 
infer_ann_file = osp.join(infer_dir, "ann_info.txt")
vis_feature = False
verbose = True