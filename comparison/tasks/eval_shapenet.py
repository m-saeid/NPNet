import re
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = re.findall('(.*)/tasks', BASE_DIR)[0]

sys.path.append(BASE_DIR)
sys.path.append(f'{BASE_DIR}/NPNet')
sys.path.append(f'{BASE_DIR}/Point_NN')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import csv
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from data.shapenet import PartNormalDataset
from time import time

from tasks.measuring_tools import *


def main(model, test_loader, data, model_name):

    print(f'\n\n  ****************************************\n  >>>>>>>>>>>>>>>>>>>> {model_name} \n  ****************************************')

    Dataset = "Shapenet"

    Time = inference_time_shapenet(model, test_loader, args.batch_size, model_name)

    GFLOPs = gflops(model, data)
    Mem = memory(model, data)  # ideally a single float in GB or MB
    Params = 0 if model_name in ['npnet', 'pointnn'] else params(model)
    # Time = inference_time_shapenet(model, test_loader, args.batch_size, model_name)
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
        "Parameters": Params,
        "InferenceTime_ms": Time,
        "NumPoints": NumPoints,
        "BatchSize": Batch_size,
        "Worker": Workers,
        "GPU_Name": gpu_name,
        "GPU_TotalMem_GB": gpu_mem_total,
        "GPU_ComputeCapability": gpu_capability
    }

    # CSV file path
    csv_file = "results.csv"
    fieldnames = list(row.keys())

    # Write to CSV (append if exists, write header if not)
    write_header = not os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("✅ Results saved to", csv_file)


def one_batch_prep(shapenet_data, model_name):

    device = 'cuda'

    data = shapenet_data.__getitem__(0)
    data = (torch.from_numpy(d).unsqueeze(0) for d in data)

    if model_name in ['npnet', 'pointnn']:
        points, shape_label, part_label, _ = data
        points = points.float().cuda().permute(0, 2, 1)
        #shape_label = shape_label.long().cuda().squeeze(1)
        #part_label = part_label.long().cuda()
        return points #, shape_label



if __name__ == '__main__':
    args = parse_args()

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    device = 'cuda'
    cudnn.benchmark = True   

    if args.model == 'npnet':
        # npnet: NPNet/models/npnet_seg.py
        from NPNet.models.npnet_seg import NPNet_Seg as npnet_model
        npnet = npnet_model(input_points=1024, num_stages=2, k_neighbors=70)
        model = npnet

    elif args.model == 'pointnn':
        # npnet: Point-NN/models/point_nn_seg.py
        from Point_NN.models.point_nn_seg import Point_NN_Seg as pointnn_model
        pointnn = pointnn_model(input_points=1024, num_stages=4)
        model = pointnn   

    else:
        raise Exception(f"MODEL NAME: {args.model}")

    # model = all_models[args.model]
    model = torch.nn.DataParallel(model)

    embd_dim = int(args.model[-2:]) if args.model[:5] == 'slnet' else None

    shapenet_data = PartNormalDataset(split='test', npoints=args.num_points, normalize=False, out_d=embd_dim)
    test_loader = DataLoader(shapenet_data, num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)

    data = one_batch_prep(shapenet_data, args.model)
        
    main(model.to(device), test_loader, data ,args.model)
