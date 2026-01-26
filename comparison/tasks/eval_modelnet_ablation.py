import re
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = re.findall('(.*)/tasks', BASE_DIR)[0]

sys.path.append(BASE_DIR)
sys.path.append(f'{BASE_DIR}/NPNet')
sys.path.append(f'{BASE_DIR}/NPNet/models')
sys.path.append(f'{BASE_DIR}/Point_NN')
sys.path.append(f'{BASE_DIR}/point_GN')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import csv
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from data.modelnet import ModelNet40
from time import time

from tasks.measuring_tools import *


def main(model, test_loader, data, model_name):

    print(f'\n\n  ****************************************\n  >>>>>>>>>>>>>>>>>>>> {model_name} \n  ****************************************')

    Dataset = "ModelNet40"
    Time = inference_time(model, test_loader, args.batch_size, model_name)
    GFLOPs = gflops(model, data)
    Mem = memory(model, data)  # ideally a single float in GB or MB
    Params = 0 if model_name in ['npnet', 'pointnn', 'pointgn'] else params(model)
    NumPoints = args.num_points
    Batch_size = args.batch_size
    Workers = args.workers

    # GPU Info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name
    gpu_capability = f"{props.major}.{props.minor}"
    gpu_mem_total = round(props.total_memory / 1e9, 2)  # in GB

    # Prepare row in your desired order
    row = {
        "Model": model_name,
        "Dataset": Dataset,
        "GFLOPs": GFLOPs,
        "GPU_Memory_Used": Mem,
        "InferenceTime_ms": Time,
        "k_Neighbors": args.k,
        "EmbedDim": args.dim,
        "NumStages": args.stage,
        "Parameters": Params,
        "NumPoints": NumPoints,
        "BatchSize": Batch_size,
        "Worker": Workers,
        "GPU_Name": gpu_name,
        "GPU_TotalMem_GB": gpu_mem_total,
        "GPU_ComputeCapability": gpu_capability
    }

    # CSV file path
    csv_file = "results_ablation.csv"
    fieldnames = list(row.keys())

    # Write to CSV (append if exists, write header if not)
    write_header = not os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("✅ Results saved to", csv_file)



if __name__ == '__main__':
    args = parse_args(dataset='modelnet')

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    device = 'cuda'
    cudnn.benchmark = True       

    if args.model == 'npnet':
        # npnet NPNet/models/npnet_cls_mn40.py; NPNet
        from NPNet.models.npnet_cls_mn40 import NPNet as npnet_model
        npnet = npnet_model(num_points=1024, init_dim=args.dim, stages=args.stage, stage_dim=args.dim, k=args.k,
                            sigma=0.232, baseline=0.1, scaling=10.0, eps=1e-6, feat_normalize=True)
        model = npnet

    elif args.model == 'pointnn':
        # pointnn PointNN/models/point_nn.py
        from Point_NN.models.point_nn import Point_NN as pointnn_model
        pointnn = pointnn_model()
        model = pointnn

    elif args.model == 'pointgn':
        # pointgn pointgn/models/point_gn.py
        from Point_GN.models.point_gn import PointGNCls as pointgn_model
        pointgn = pointgn_model(init_dim=72, stages=4, stage_dim=72, sigma=0.3, feat_normalize=True)
        model = pointgn
    
    else:
        raise Exception("MODEL NAME: {args.model}")

    # model = all_models[args.model]
    model = torch.nn.DataParallel(model)

    embd_dim = int(args.model[-2:]) if args.model[:5] == 'slnet' else None
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, in_d=3, out_d=embd_dim), num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)

    data = torch.randn(1, 1024, 3).cuda().contiguous() if args.model in ['npnet', 'pointgn'] else torch.randn(1, 3, 1024).cuda().contiguous()
    if args.model[:5] == 'slnet':
        feat = torch.randn(1, embd_dim, 1024).cuda().contiguous()
        data = (data, feat)
        
    main(model.to(device), test_loader, data ,args.model)