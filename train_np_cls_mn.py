import os
import time
import csv
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F

import general_utils as gutils
from models.npnet_cls_mn40 import NPNet


def create_log_file(task_type, dataset):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("results", "logs", task_type, dataset)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{now}.txt")
    return log_path


def process_data(data_loader, model, device):
    features_list, labels_list = [], []
    for points, labels in tqdm(data_loader, leave=False):
        point_features = model(points.to(device))
        features_list.append(point_features)
        labels = labels.to(device)
        labels_list.append(labels)
    features = torch.cat(features_list, dim=0)
    features = F.normalize(features, dim=-1)
    labels = torch.cat(labels_list, dim=0)
    return features, labels


def get_model_and_param(args):
    model = NPNet(
        num_points=args.num_points,
        init_dim=args.init_dim,
        stages=args.stages,
        stage_dim=args.stage_dim,
        k=args.k,
        feat_normalize=args.feat_normalize,
        sigma=args.sigma,
        baseline=args.baseline, #0.1,
        scaling=args.scaling,   #10.0,
        eps=args.eps,           # 1e-6
        fixed_sigma=args.fixed_sigma, # None
        fixed_blend=args.fixed_blend  # None
    )
    return model


def process_and_evaluate(train_loader, test_loader, model, device):
    start_train_time = time.time()
    train_features, train_labels = process_data(train_loader, model, device)
    train_labels = F.one_hot(train_labels).squeeze().float()
    train_time = time.time() - start_train_time

    start_test_time = time.time()
    test_features, test_labels = process_data(test_loader, model, device)
    test_time = time.time() - start_test_time

    acc_cos, gamma = gutils.cosine_similarity(test_features, train_features, train_labels, test_labels)
    acc_1nn = gutils.one_nn_classification(test_features, train_features, train_labels, test_labels)

    return {
        "train_time": train_time,
        "test_time": test_time,
        "acc_cos": acc_cos,
        "acc_1nn": acc_1nn,
        "gamma": gamma,
    }


@torch.no_grad()
def main_cls():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.abspath(os.path.join(current_dir, ".."))
    dataset_dir = os.path.join(current_dir, "datasets")

    args = gutils.get_arguments()
    train_loader, test_loader = gutils.get_cls_dataloader(dataset_dir, args)

    gutils.set_seed(args.seed)

    # Determine results folder and CSV filename.
    
    results_folder = os.path.join(current_dir, "results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if args.dataset == "modelnet40":
        csv_filename = os.path.join(results_folder, "npnet_cls_modelnet40.csv")
        mode_val = ""
    if args.dataset == "modelnet-R":
        csv_filename = os.path.join(results_folder, "npnet_cls_modelnetR.csv")
        mode_val = ""
    elif args.dataset == "modelnet40fewshot":
        csv_filename = os.path.join(results_folder, f"npnet_fewshot_modelnet.csv")
        mode_val = f"{args.n_way}_way {args.k_shots}_shots"

    # csv_filename = f"results/cls_{args.dataset}.csv"

    # Check if this configuration has already been executed.
    if os.path.exists(csv_filename):
        with open(csv_filename, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Compare keys. For embedding functions that don't use sigma, we require sigma to match a default value (e.g. 0)
                row_sigma = float(row["sigma"])
                current_sigma = args.sigma

                if args.fixed_sigma is None and row["fixed_sigma"] == "None":
                    fixed_sigma_ = 0
                    current_fixed_sigma_ = 0
                elif args.fixed_sigma is not None and row["fixed_sigma"] != "None":
                    fixed_sigma_ = float(row["fixed_sigma"])
                    current_fixed_sigma_ = args.fixed_sigma
                else:
                    continue  # One is None, the other is not.

                if args.fixed_blend is None and row["fixed_blend"] == "None":
                    fixed_blend_ = 0
                    current_fixed_blend_ = 0
                elif args.fixed_blend is not None and row["fixed_blend"] != "None":
                    fixed_blend_ = float(row["fixed_blend"])
                    current_fixed_blend_ = args.fixed_blend
                else:
                    continue  # One is None, the other is not.

                if (
                    # row["model"] == args.model and
                    row["dataset"] == args.dataset and
                    row["mode"] == mode_val and
                    int(row["seed"]) == args.seed and
                    int(row["k"]) == args.k and
                    int(row["init_dim"]) == args.init_dim and
                    int(row["stages"]) == args.stages and
                    int(row["stage_dim"]) == args.stage_dim and
                    abs(row_sigma - current_sigma) < 1e-6 and
                    abs(fixed_sigma_ - current_fixed_sigma_) < 1e-6 and
                    abs(fixed_blend_ - current_fixed_blend_) < 1e-6
                ):
                    print("Configuration already executed. Skipping this run.")
                    return  # Exit without re-running the experiment.

    print("Configuration not found in CSV; proceeding with training...")

    model = get_model_and_param(args)
    model.to(device).eval()
    start_total_time = time.time()

    if args.dataset == "modelnet40fewshot":
        train_times = []
        test_times = []
        acc_cos = []
        acc_1nn = []
        for fold in tqdm(range(10)):
            train_loader.dataset.set_fold(fold)
            test_loader.dataset.set_fold(fold)
            results = process_and_evaluate(train_loader, test_loader, model, device)
            train_times.append(results["train_time"])
            test_times.append(results["test_time"])
            acc_cos.append(results["acc_cos"])
            acc_1nn.append(results["acc_1nn"])
        results = {
            "train_time": np.mean(train_times),
            "test_time": np.mean(test_times),
            "acc_cos": np.mean(acc_cos),
            "acc_1nn": np.mean(acc_1nn),
            "gamma": 0,
        }
    else:
        results = process_and_evaluate(train_loader, test_loader, model, device)

    total_time = time.time() - start_total_time

    print("==============================")
    print("model = {}".format(args.model))
    print("dataset = {}".format(args.dataset))
    print("mode = {}".format(mode_val))
    print("seed = {}".format(args.seed))
    print("k = {}".format(args.k))
    print("init_dim (idim) = {}".format(args.init_dim))
    print("stages = {}".format(args.stages))
    print("stage_dim (fdim) = {}".format(args.stage_dim))
    print("sigma = {}".format(args.sigma))
    print("eps = {}".format(args.eps))
    print("total_time = {}".format(round(total_time,2)))
    print("train_time = {}".format(round(results['train_time'],2)))
    print("test_time = {}".format(round(results['test_time'],2)))
    print("acc_cos = {}".format(round(results['acc_cos'],2)))
    print("acc_1nn = {}".format(round(results['acc_1nn'],2)))
    print("gamma = {}".format(round(results['gamma'],2)))


    # Create log file
    log_file_path = create_log_file("classification", args.dataset)

    # Write detailed log
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Classification Log - {args.dataset}\n")
        log_file.write(f"Model: {args.model}\n")
        log_file.write(f"Accuracy (cos): {results['acc_cos']}\n")
        log_file.write(f"Accuracy (1-NN): {results['acc_1nn']}\n")
        log_file.write(f"Total Time: {total_time:.2f}s\n")
        log_file.write(f"Train Time: {results['train_time']:.2f}s\n")
        log_file.write(f"Test Time: {results['test_time']:.2f}s\n")
        log_file.write(f"Args: {vars(args)}\n")


    if args.evaluation:
        from measuring_tools import inference_time, gflops, memory
        
        if args.dataset == "modelnet40":
            from data.modelnet40 import ModelNet40
            from torch.utils.data import DataLoader
            data = torch.randn(1, 1024, 3).cuda().contiguous()
            test_loader = DataLoader(ModelNet40(dataset_dir=dataset_dir, partition='test', num_points=args.num_points), num_workers=8,
                                    batch_size=args.batch_size, shuffle=False, drop_last=False)
            Time = inference_time(model, test_loader, args.batch_size, model_name="npnet")
            GFLOPs = gflops(model, data)
            Mem = memory(model, data)  # ideally a single float in GB or MB
            print(f"Avg Inference Time per Batch: {Time:.4f},\nGFLOPs: {GFLOPs:.4f},\nMem: {Mem:.4f} MB")

        # Append the results to the CSV file.
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = [
                "dataset", "acc_cos", "acc_1nn", "Inference_Time", "GFLOPs", "Mem_MB", "mode", "seed", "k", "init_dim",
                "stages", "stage_dim", "sigma", "baseline", "scaling", "eps", "fixed_sigma", "fixed_blend",
                "batch_size", "num_points",
                "total_time", "train_time", "test_time", "gamma", "log_file_path"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "dataset": args.dataset,
                "acc_cos": results["acc_cos"],
                "acc_1nn": results["acc_1nn"],

                "Inference_Time": Time,
                "GFLOPs": GFLOPs,
                "Mem_MB": Mem,

                "mode": mode_val,
                "seed": args.seed,
                "k": args.k,
                "init_dim": args.init_dim,
                "stages": args.stages,
                "stage_dim": args.stage_dim,
                "sigma": args.sigma,
                "baseline": args.baseline,
                "scaling": args.scaling,
                "eps": args.eps,

                "fixed_sigma": "None" if args.fixed_sigma is None else args.fixed_sigma,
                "fixed_blend": "None" if args.fixed_blend is None else args.fixed_blend,

                "batch_size": args.batch_size,
                "num_points": args.num_points,
                "total_time": total_time,
                "train_time": results["train_time"],
                "test_time": results["test_time"],
                "gamma": results["gamma"],
                "log_file_path": log_file_path
            })
    else:
        print("Skipping evaluation")
        # Append the results to the CSV file.
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = [
                "dataset", "acc_cos", "acc_1nn", "Inference_Time", "GFLOPs", "Mem_MB", "mode", "seed", "k", "init_dim",
                "stages", "stage_dim", "sigma", "baseline", "scaling", "eps", "fixed_sigma", "fixed_blend",
                "batch_size", "num_points",
                "total_time", "train_time", "test_time", "gamma", "log_file_path"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "dataset": args.dataset,
                "acc_cos": results["acc_cos"],
                "acc_1nn": results["acc_1nn"],

                "Inference_Time": "-",
                "GFLOPs": "-",
                "Mem_MB": "-",

                "mode": mode_val,
                "seed": args.seed,
                "k": args.k,
                "init_dim": args.init_dim,
                "stages": args.stages,
                "stage_dim": args.stage_dim,
                "sigma": args.sigma,
                "baseline": args.baseline,
                "scaling": args.scaling,
                "eps": args.eps,

                "fixed_sigma": "None" if args.fixed_sigma is None else args.fixed_sigma,
                "fixed_blend": "None" if args.fixed_blend is None else args.fixed_blend,

                "batch_size": args.batch_size,
                "num_points": args.num_points,
                "total_time": total_time,
                "train_time": results["train_time"],
                "test_time": results["test_time"],
                "gamma": results["gamma"],
                "log_file_path": log_file_path
            })

if __name__ == "__main__":
    main_cls()
