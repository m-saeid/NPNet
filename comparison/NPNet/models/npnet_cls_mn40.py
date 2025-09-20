import os
import sys
import torch
import torch.nn as nn

project_path = os.path.abspath(".")
sys.path.append(project_path)
print(project_path)

from NPNet.models.model_utils import AdaptiveEncoderCls


class NPNet(nn.Module):
    def __init__(self, num_points=1024, init_dim=6, stages=4, stage_dim=72, k=90, sigma=0.3, baseline=0.1, scaling=10.0, eps=1e-6, feat_normalize=True, fixed_sigma=None, fixed_blend=None, **kwargs):
        super().__init__()
        self.AdaEnc = AdaptiveEncoderCls(num_points, init_dim, stages, stage_dim, k, sigma, baseline, scaling, eps, feat_normalize, fixed_sigma, fixed_blend, **kwargs)

    def forward(self, xyz):
        x = self.AdaEnc(xyz)
        return x


if __name__ == "__main__":
    import time
    batch_size = 128
    num_points = 1024
    in_ch = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    random_xyz = torch.randn(batch_size, num_points, in_ch).contiguous().to(device)
    model = NPNet(num_points=num_points).to(device)
    start_time = time.time()
    output = model(random_xyz)
    end_time = time.time()
    print(f"Output shape: {output.shape}")
    print(f"Time taken for forward pass: {end_time - start_time:.6f} seconds")
