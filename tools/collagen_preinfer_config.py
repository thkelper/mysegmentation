import os.path as osp

# infer = True
infer = False

# data_root_dir = "/home/yangchangpeng/wing_studio/data/col_dataset"
data_root_dir = "/home/yangchangpeng/wing_studio/data/collagen_data"
img_dir = osp.join(data_root_dir, "0806_2230_48well_col1d_cell1000k_p")
infer_dir = osp.join(data_root_dir, "infer_0806_2230_48well_col1d_cell1000k_p")

model_cfg_path = "./results/hrnet/fcn_hr18_4xb2-40k_cityscapes-512x1024.py"
model_ckpt_path = "./results/hrnet/best_mIoU_iter_5104.pth"
IMG_EXT = ".bmp"
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
end_idx = 2

# infer_configs 
infer_ann_file = osp.join(infer_dir, "ann_info.txt")
vis_feature = False
verbose = True