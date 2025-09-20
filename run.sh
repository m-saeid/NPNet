#!/bin/bash

# Classification - ModelNet40
python train_np_cls_mn.py --dataset modelnet40                                  # acc: 85.45
python train_np_cls_mn.py --dataset modelnet-R                                  # acc: 85.65

# Classification - ModelNet40 - Few Shot Learning
python train_np_cls_mn.py --dataset modelnet40fewshot --n_way 5 --k_shots 10    # acc:92.0
python train_np_cls_mn.py --dataset modelnet40fewshot --n_way 5 --k_shots 20    # acc:93.2
python train_np_cls_mn.py --dataset modelnet40fewshot --n_way 10 --k_shots 10   # acc:82.45
python train_np_cls_mn.py --dataset modelnet40fewshot --n_way 10 --k_shots 20   # acc:87.55

# Classification - ScanObject
python train_np_cls_scan.py --split OBJ_BG                                      # 86.1
python train_np_cls_scan.py --split OBJ_ONLY                                    # 86.1
python train_np_cls_scan.py --split PB_T50_RS                                   # 84.9

# Segmentation - ShapeNet
python train_np_seg.py --dataset shapenetpart                                   # acc: 73.5