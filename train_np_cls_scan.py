import os
import csv
import time
import torch
import torch.nn.functional as F
import general_utils_so as gutils
from models.model_cls_scan import NPNet
from models.model_utils import process_and_evaluate


@torch.no_grad()
def main_cls():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    dataset_dir = os.path.join(project_root, "datasets")

    args = gutils.get_arguments()
    train_loader, test_loader = gutils.get_cls_dataloader(dataset_dir, args)

    gutils.set_seed(args.seed)

    model = NPNet(
        num_points=args.num_points,
        init_dim=args.init_dim,
        stages=args.stages,
        stage_dim=args.stage_dim,
        k=args.k,
        sigma=args.sigma,
        feat_normalize=args.feat_normalize,
        )

    model.to(device).eval()
    start_total_time = time.time()

    results = process_and_evaluate(train_loader, test_loader, model, device)

    total_time = time.time() - start_total_time

    print("==============================")
    print(f"model = {args.model}")
    print(f"dataset = {args.dataset}")
    print(f"seed = {args.seed}")
    print(f"k = {args.k}")
    print(f"init_dim = {args.init_dim}")
    print(f"stages = {args.stages}")
    print(f"stage_dim = {args.stage_dim}")
    print(f"total_time = {total_time}")
    print(f"train_time = {results['train_time']}")
    print(f"test_time = {results['test_time']}")
    print(f"acc_cos = {results['acc_cos']}")
    print(f"acc_1nn = {results['acc_1nn']}")
    print(f"gamma = {results['gamma']}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(current_dir, "results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    csv_filename = os.path.join(results_folder, f"npnet_cls_scanobject_{args.split}.csv")
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = [
            "dataset", "acc_cos", "acc_1nn", "seed", "k", "sigma", "init_dim", "stage_dim", "stages", "total_time", "train_time", "test_time", "gamma"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "dataset": args.dataset,
            "acc_cos": results['acc_cos'],
            "acc_1nn": results['acc_1nn'],
            "seed": args.seed,
            "k": args.k,
            "sigma": args.sigma,
            "init_dim": args.init_dim,
            "stage_dim": args.stage_dim,
            "stages": args.stages,
            "total_time": total_time,
            "train_time": results['train_time'],
            "test_time": results['test_time'],
            "gamma": results['gamma'],
        })



if __name__ == "__main__":
    main_cls()
