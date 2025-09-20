import os
import csv
import torch
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import argparse
import random

from data.shapenet import PartNormalDataset
from models.npnet_seg import NPNet_Seg

from measuring_tools import *


def create_log_file(task_type, dataset):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("results", "logs", task_type, dataset)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{now}.txt")
    return log_path


def compute_overall_iou(pred, target, num_classes=50):
    shape_ious = []
    pred = pred.max(dim=2)[1]   # (batch_size, num_points)  the pred_class_idx of each point in each sample
    pred_np = pred.cpu().data.numpy()

    target_np = target.cpu().data.numpy()
    
    for shape_idx in range(pred.shape[0]):   # sample_idx
        part_ious = []
        for part in range(num_classes):   # class_idx! no matter which category, only consider all part_classes of all categories, check all 50 classes
            # for target, each point has a class no matter which category owns this point! also 50 classes!!!
            # only return 1 when both belongs to this class, which means correct:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            # always return 1 when either is belongs to this class:
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            F = np.sum(target_np[shape_idx] == part)

            if F != 0:       
                iou = I / float(U)    #  iou across all points for this class
                part_ious.append(iou)   #  append the iou of this class
        shape_ious.append(np.mean(part_ious))   # each time append an average iou across all classes of this sample (sample_level!)
    return shape_ious   # [batch_size]

def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='shapenetpart')  # 71.27, 73.95

    parser.add_argument('--bz', type=int, default=1)  # Freeze as 1

    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--stages', type=int, default=2)
    parser.add_argument('--dim', type=int, default=144)
    parser.add_argument('--k', type=int, default=70)
    parser.add_argument('--de_k', type=int, default=6)  # propagate neighbors in decoder
    parser.add_argument('--alpha', type=int, default=1000)
    parser.add_argument('--beta', type=int, default=100)

    # Adaptive
    parser.add_argument("--sigma", type=float, default=0.6729, help="sigma value used by embedding function")     #
    parser.add_argument("--baseline", type=float, default=0.1, help="baseline value used by embedding function") #
    parser.add_argument("--scaling", type=float, default=10.0, help="scaling value used by embedding function")  #
    parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon value for numerical stability")         #

    parser.add_argument('--encoder_type', type=str, default='seg')
    parser.add_argument('--adaptive_ratio', type=float, default=0.5)

    parser.add_argument('--gamma', type=int, default=300)  # Best as 300

    parser.add_argument('--seed', type=int, default=26)

    parser.add_argument("--evaluation", action="store_true", default=False, help="Evaluation [True/False]")


    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def main():

    print('==> Loading args..')
    args = get_arguments()
    print(args)

    set_seed(args.seed)

    # Determine results folder and CSV filename.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "datasets", "shapenetcore_partanno_segmentation_benchmark_v0_normal")

    # Determine results folder and CSV filename.
    results_folder = os.path.join(current_dir, "results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    csv_filename = os.path.join(results_folder, "npnet_seg_shapenet.csv")

    # Check if this configuration has already been executed.
    if os.path.exists(csv_filename):
        with open(csv_filename, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Compare keys. For embedding functions that don't use sigma, we require sigma to match a default value (e.g. 0)
                row_sigma = float(row["sigma"])
                current_sigma = args.sigma
                row_baseline = float(row["baseline"])
                current_baseline = args.baseline
                row_scaling = float(row["scaling"])
                current_scaling = args.scaling
                row_adaptive_ratio = float(row["adaptive_ratio"])
                current_adaptive_ratio = args.adaptive_ratio
                if (
                    row["dataset"] == args.dataset and
                    row["encoder_type"] == args.encoder_type and

                    int(row["bz"]) == args.bz and
                    int(row["points"]) == args.points and
                    int(row["stages"]) == args.stages and
                    int(row["dim"]) == args.dim and
                    int(row["k"]) == args.k and
                    int(row["de_k"]) == args.de_k and
                    int(row["alpha"]) == args.alpha and
                    int(row["beta"]) == args.beta and
                    # int(row["scaling"]) == args.scaling and
                    int(row["k"]) == args.k and
                    int(row["gamma"]) == args.gamma and
                    int(row["seed"]) == args.seed and
                    abs(row_sigma - current_sigma) < 1e-6 and
                    abs(row_scaling - current_scaling) < 1e-6 and
                    abs(row_baseline - current_baseline) < 1e-6 and
                    abs(row_adaptive_ratio - current_adaptive_ratio) < 1e-6
                ):
                    print("Configuration already executed. Skipping this run.")
                    return  # Exit without re-running the experiment.  #########################################################

    print("Configuration not found in CSV; proceeding with training...")


    print('==> Preparing model..')
    npnet_Seg = NPNet_Seg(input_points=args.points, num_stages=args.stages,
                            embed_dim=args.dim, k_neighbors=args.k, de_neighbors=args.de_k,
                            alpha=args.alpha, beta=args.beta,
                            sigma=args.sigma, baseline=args.baseline, scaling=args.scaling, eps=args.eps,
                            encoder_type=args.encoder_type, adaptive_ratio=args.adaptive_ratio).cuda()

    npnet_Seg.eval()

    print('==> Preparing data..')
    train_loader = DataLoader(PartNormalDataset(npoints=args.points, split='trainval', normalize=False, dataset_dir=dataset_dir), 
                                num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
    test_loader = DataLoader(PartNormalDataset(npoints=args.points, split='test', normalize=False, dataset_dir=dataset_dir), 
                                num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)


    print('==> Constructing Point-Memory Bank..')
    num_part, num_shape = 50, 16
    # We organize point-memory bank by 16 shape labels
    feature_memory = [[] for i in range(num_shape)]
    label_memory = [[] for i in range(num_shape)]

    for points, shape_label, part_label, norm_plt in tqdm(train_loader):
    
        # pre-process
        points = points.float().cuda().permute(0, 2, 1)
        shape_label = shape_label.long().cuda().squeeze(1)
        part_label = part_label.long().cuda()

        # Pass through the Non-Parametric Encoder + Decoder
        point_features = npnet_Seg(points)
        # All 2048 point features in a shape
        point_features = point_features.permute(0, 2, 1)  # bz, 2048, c

        # Extracting part prototypes for a shape
        feature_memory_list = []
        label_memory_list = []

        for i in range(num_part):
            # Find the point indices for the part_label within a shape
            part_mask = (part_label == i)
            if torch.sum(part_mask) == 0:
                continue
            # Extract point features for the part_label
            part_features = point_features[part_mask]
            # Obtain part prototypes by average point features for the part_label
            part_features = part_features.mean(0).unsqueeze(0)
            
            feature_memory_list.append(part_features)
            label_memory_list.append(torch.tensor(i).unsqueeze(0))
        
        # Feature Memory: store prototypes indexed by the corresponding shape_label
        feature_memory_list = torch.cat(feature_memory_list, dim=0)
        feature_memory[int(shape_label)].append(feature_memory_list)

        # Label Memory: store labels indexed by the corresponding shape_label
        label_memory_list = torch.cat(label_memory_list, dim=0)
        label_memory_list = F.one_hot(label_memory_list, num_classes=num_part)
        label_memory[int(shape_label)].append(label_memory_list)

    # Organize the point-memory bank
    for i in range(num_shape):
        # Feature Memory
        feature_memory[i] = torch.cat(feature_memory[i], dim=0)
        feature_memory[i] /= feature_memory[i].norm(dim=-1, keepdim=True)
        feature_memory[i] = feature_memory[i].permute(1, 0)
        print("Feature Memory of the " + str(i) + "-th shape is", feature_memory[i].shape)
        # Label Memory
        label_memory[i] = torch.cat(label_memory[i], dim=0).cuda().float()


    print('==> Starting NPNet..')
    logits_list, label_list = [], []
    for points, shape_label, part_label, norm_plt in tqdm(test_loader):

        # pre-process
        points = points.float().cuda().permute(0, 2, 1)
        shape_label = shape_label.long().cuda().squeeze(1)
        part_label = part_label.long().cuda()

        # Pass through the Non-Parametric Encoder + Decoder
        point_features = npnet_Seg(points)
        point_features = point_features.permute(0, 2, 1).squeeze(0)  # 2048, c
        point_features /= point_features.norm(dim=-1, keepdim=True)
        
        # Similarity Matching
        Sim = point_features @ feature_memory[int(shape_label)]

        # Label Integrate
        logits = (-args.gamma * (1 - Sim)).exp() @ label_memory[int(shape_label)]
  
        logits_list.append(logits.unsqueeze(0))
        label_list.append(part_label)
            
    logits_list = torch.cat(logits_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    # Compute mIoU
    iou = compute_overall_iou(logits_list, label_list)
    miou = np.mean(iou) * 100
    
    print(f"Part Segmentation mIoU: {miou:.2f}.")


    # Create log file
    log_file_path = create_log_file("segmentation", args.dataset)

    # Write detailed log
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Segmentation Log - {args.dataset}\n")
        # log_file.write(f"IOU: {iou:.4f}, MIOU: {miou:.4f}\n")
        log_file.write(f"MIOU: {miou}, IOU: {str(iou)}\n")
        log_file.write(f"Args: {vars(args)}\n")


    if args.evaluation:
        

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


        if args.dataset == "shapenetpart":
            args.batch_size = 16 # args.bz
            shapenet_data = PartNormalDataset(split='test', npoints=args.points, normalize=False, dataset_dir=dataset_dir)
            test_loader = DataLoader(shapenet_data, num_workers=8,
                                    batch_size=args.batch_size, shuffle=False, drop_last=False)

            data = one_batch_prep(shapenet_data, 'npnet')

            Time = inference_time_shapenet(npnet_Seg, test_loader, args.batch_size, model_name="npnet")
            GFLOPs = gflops(npnet_Seg, data)
            Mem = memory(npnet_Seg, data)  # ideally a single float in GB or MB
            print(f"Avg Inference Time per Batch: {Time:.4f},\nGFLOPs: {GFLOPs:.4f},\nMem: {Mem:.4f} MB")


        # Append the results to the CSV file.
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = [
                "dataset", "miou", "iou", "Inference_Time", "GFLOPs", "Mem_MB", "encoder_type", "bz", "points", "stages",
                "dim", "k", "de_k", "alpha", "beta", "sigma", "baseline", "scaling", "eps",
                "adaptive_ratio", "gamma", "seed", "log_file_path"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "dataset": args.dataset,
                "miou":miou,
                "iou":"", #str(iou),
                
                "Inference_Time": Time,
                "GFLOPs": GFLOPs,
                "Mem_MB": Mem,

                "encoder_type": args.encoder_type,
                "bz": args.bz,
                "points": args.points,
                "stages": args.stages,
                "dim": args.dim,
                "k": args.k,
                "de_k": args.de_k,
                "alpha": args.alpha,
                "beta": args.beta,
                "sigma": args.sigma,
                "baseline": args.baseline,
                "scaling": args.scaling,
                "eps": args.eps,

                "adaptive_ratio": args.adaptive_ratio,

                "gamma": args.gamma,
                "seed": args.seed,
                "log_file_path": log_file_path,
            })

    else:
        print("Evaluation flag not set; skipping performance measurement.")
                # Append the results to the CSV file.
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = [
                "dataset", "miou", "iou", "Inference_Time", "GFLOPs", "Mem_MB", "encoder_type", "bz", "points", "stages",
                "dim", "k", "de_k", "alpha", "beta", "sigma", "baseline", "scaling", "eps",
                "adaptive_ratio", "gamma", "seed", "log_file_path"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "dataset": args.dataset,
                "miou":miou,
                "iou":"", #str(iou),
                
                "Inference_Time": "-",
                "GFLOPs": "-",
                "Mem_MB": "-",

                "encoder_type": args.encoder_type,
                "bz": args.bz,
                "points": args.points,
                "stages": args.stages,
                "dim": args.dim,
                "k": args.k,
                "de_k": args.de_k,
                "alpha": args.alpha,
                "beta": args.beta,
                "sigma": args.sigma,
                "baseline": args.baseline,
                "scaling": args.scaling,
                "eps": args.eps,

                "adaptive_ratio": args.adaptive_ratio,

                "gamma": args.gamma,
                "seed": args.seed,
                "log_file_path": log_file_path,
            })

if __name__ == '__main__':
    main()