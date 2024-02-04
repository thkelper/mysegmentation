# -*- coding: UTF-8 -*-
import os
import os.path as osp
import shutil
from collections import defaultdict
from pprint import pprint

def osp_join(*paths):
    return os.path.join(*paths)

def get_fpath_list(dir_path, valid_ext=None, recursive=False, ret_fnames=False, verbose=False):
    paths = list()
    invalid_paths = list()
    fnames = list()
    if not recursive:
        assert osp.isdir(dir_path), "must input a folder path."
    if recursive:
        total_cnt = 0
        for root, dirs, files in os.walk(dir_path):
            for fname in files:
                fpath = osp_join(root, fname)
                if valid_ext:
                    if fname.split(".")[-1] in valid_ext:
                        fnames.append(fname)
                        paths.append(fpath)
                    elif fname.split(".")[-1] not in valid_ext:
                        invalid_paths.append(fpath)
                else:
                    fnames.append(fname)
                    paths.append(fpath)
                total_cnt += 1
        print(f"Total {total_cnt} paths, valid:{len(paths)} invalid:{len(invalid_paths)}")            
    else:
        for i, fname in enumerate(sorted(os.listdir(dir_path))):
            fpath = osp_join(dir_path, fname)
            # import pdb
            # pdb.set_trace()
            if valid_ext:
                if fname.split(".")[-1] in valid_ext:
                    fnames.append(fname)
                    paths.append(fpath)
                elif fname.split(".")[-1] not in valid_ext:
                    invalid_paths.append(fpath)
            else:
                fnames.append(fname)
                paths.append(fpath)
        print(f"Total {i+1} paths, valid:{len(paths)}, invalid:{len(invalid_paths)}.")
    if len(invalid_paths) and not verbose:
        print(f"You can set verbose=True to check those invalid paths!")
    elif len(invalid_paths) and verbose:
        pprint(invalid_paths)
    if ret_fnames:
        return paths, fnames
    return paths

def get_fname_list(dir_path):
    names = list()
    assert osp.isdir(dir_path), "must input a folder."
    for fname in sorted(os.listdir(dir_path)):
        names.append(fname)
    print(f"Total {len(names)} file names!")
    return names


def mkrs(path, rm_exists = False):
    # print(f"ycp test {path}")
    # os.makedirs(path, exist_ok=True)
    if os.path.exists(path):
        if not rm_exists:
            # print(f'文件夹{path}已存在，且默认不进行删除')
            print(f"{path} exists")
        else:
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path)


def print_coco_stat(coco):
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']

    print(classes)
    class_objs = defaultdict(int)
    total_objs = 0
    imgs_id = coco.getImgIds()
    size_stat = {20 * i: 0 for i in range(1, 6)}
    sizes = list(size_stat.keys())
    sizes.sort()
    sizes_num = len(size_stat.keys())

    for img_id in imgs_id:
        img = coco.loadImgs(img_id)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            name = classes[ann['category_id']]
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = (int)(bbox[0])
                ymin = (int)(bbox[1])
                xmax = (int)(bbox[2] + bbox[0])
                ymax = (int)(bbox[3] + bbox[1])
                size_stat[sizes[min(sizes_num - 1, (ymax - ymin) * (xmax - xmin) // sizes[0])]] += 1

            class_objs[name] += 1
            total_objs += 1

    total_imgs = len(imgs_id)
    total_class = len(classes)
    print('total imgs: {}, total class: {}, total objects: {}'.format(total_imgs, total_class, total_objs))
    print(size_stat)
    print(class_objs)



if __name__ == "__main__":
    get_fpath_list("/data/datasets/cv/humujing_dataset/images", valid_ext=".png", recursive=True)
