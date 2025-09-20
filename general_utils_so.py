import os
import csv
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.scanobjectnn import ScanObjectNN

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="NPNet")
    parser.add_argument("--task", type=str, default="cls")
    parser.add_argument("--dataset", type=str, default="scanobject")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--init-dim", type=int, default=27)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--stage-dim", type=int, default=27)
    parser.add_argument("--k", type=int, default=120)

    parser.add_argument(
        "--split",
        type=str,
        default="OBJ_BG",
        choices=["OBJ_BG", "OBJ_ONLY", "PB_T50_RS"],
    )
    parser.add_argument("--feat_normalize", action="store_true", default=False)
    return parser

def additional_dataset_arguments(parser, split):
    if split == "OBJ_BG":
        parser.add_argument("--sigma", type=float, default=0.137)
        parser.add_argument("--seed", type=int, default=42)
    elif split == "OBJ_ONLY":
        parser.add_argument("--sigma", type=float, default=0.1876)
        parser.add_argument("--seed", type=int, default=42)
    elif split == "PB_T50_RS":
        parser.add_argument("--sigma", type=float, default=0.112)
        parser.add_argument("--seed", type=int, default=193)

def get_arguments():
    parser = setup_parser()
    args, _ = parser.parse_known_args()
    additional_dataset_arguments(parser, args.split)
    args = parser.parse_args()
    args.csv_file = generate_csv_filename(args)
    return args


def generate_csv_filename(args):
    csv_file = f"{args.model}_{args.task}_{args.dataset}_n{args.feat_normalize}"
    if args.dataset == "scanobject":
        csv_file += f"_s{args.split}"
    return os.path.join("./eval", csv_file + ".csv")


def get_dataloader(dataset, num_points, batch_size, partition, **kwargs):
    num_workers = 8
    return DataLoader(
        dataset(num_points=num_points, partition=partition, **kwargs),
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


def get_cls_dataloader(dataset_dir, args):
    dataset_dir = os.path.join(dataset_dir, "scanobjectnn")
    kwargs = {"split": args.split}

    train_loader = get_dataloader(ScanObjectNN, args.num_points, args.batch_size, "train", dataset_dir=dataset_dir, **kwargs)
    test_loader = get_dataloader(ScanObjectNN, args.num_points, args.batch_size, "test", dataset_dir=dataset_dir, **kwargs)

    return train_loader, test_loader


def load_csv_as_dict(csv_file):
    """Loads the CSV file if it exists, or returns an empty list if the file doesn't exist."""
    if os.path.exists(csv_file):
        int_fields = ["batch", "seed", "idim", "fdim", "stage", "k"]
        float_fields = ["sigma"]

        with open(csv_file, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            data = []
            for row in reader:
                for field in int_fields:
                    if field in row and row[field] != "":
                        row[field] = int(row[field])  # Convert to integer
                for field in float_fields:
                    if field in row and row[field] != "":
                        row[field] = float(row[field])  # Convert to float
                data.append(row)
            return data
    else:
        return []


def check_if_combination_exists(
    data, batch, stage, sigma, seed, init_dim, stage_dim, k
):
    """Checks if the given combination has already been calculated in the data."""
    for entry in data:
        if (
            entry["batch"] == batch
            and entry["sigma"] == sigma
            and entry["seed"] == seed
            and entry["idim"] == init_dim
            and entry["fdim"] == stage_dim
            and entry["stage"] == stage
            and entry["k"] == k
        ):
            # print(
            #     f"Skipping already processed combination: sigma={sigma}, seed={seed}, fdim={stage_dim}, stage={stage}, k={k}"
            # )
            return True
    return False


def add_new_entry(data, batch, combo, acc_1nn, acc_cos, gamma, train_time, test_time):
    stage, sigma, seed, init_dim, stage_dim, k = combo

    new_entry = {
        "batch": batch,
        "sigma": sigma,
        "seed": seed,
        "idim": init_dim,
        "fdim": stage_dim,
        "stage": stage,
        "k": k,
        "acc_1nn": acc_1nn,
        "acc_cos": acc_cos,
        "gamma": gamma,
        "train_time": train_time,
        "test_time": test_time,
    }
    data.append(new_entry)
    # print(f"Added new entry: {new_entry}")


def save_data_to_csv(data, filename):

    headers = data[0].keys() if data else []
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
    # print(f"Data saved to {filename}")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_similarity(test_features, train_features, train_labels, test_labels):
    # test_features [num_test_samples, num_features],
    # train_features [num_train_samples, num_features]

    test_features = F.normalize(test_features, dim=-1)
    train_features = F.normalize(train_features, dim=-1)

    gamma_list = [i * 10000 / 5000 for i in range(5000)]
    best_gamma_acc, best_gamma = 0, 0

    # Loop through each gamma value to find the one that gives the best accuracy
    for gamma in tqdm(gamma_list, leave=False):
        sim = test_features @ train_features.t()
        logits = torch.exp(-gamma * (1 - sim)) @ train_labels
        pred = logits.topk(1, dim=1, largest=True, sorted=True).indices
        correct = pred.squeeze().eq(test_labels.view(-1))
        acc = correct.float().mean().item() * 100

        if acc > best_gamma_acc:
            best_gamma_acc, best_gamma = acc, gamma

    return best_gamma_acc, best_gamma


def one_nn_classification(test_features, train_features, train_labels, test_labels):

    # Ensure train_labels are class indices
    if train_labels.ndim > 1:
        # Assuming one-hot encoding
        train_labels = train_labels.argmax(dim=1)

    # Normalize the feature vectors to unit length
    test_features = F.normalize(test_features, p=2, dim=1)
    train_features = F.normalize(train_features, p=2, dim=1)

    # Compute cosine similarity between test and train features
    # [num_test_samples, num_train_samples]
    similarity = torch.mm(test_features, train_features.t())

    # For each test sample, find the index of the nearest train sample
    # Since features are normalized, cosine similarity is equivalent to dot product
    # Higher similarity means closer
    nearest_indices = similarity.argmax(dim=1)  # [num_test_samples]

    # Retrieve the predicted labels from the nearest neighbors
    pred_labels = train_labels[nearest_indices]

    # Compare predictions with true labels
    correct = pred_labels.eq(test_labels)

    # Calculate accuracy
    accuracy = correct.float().mean().item() * 100.0

    return accuracy
