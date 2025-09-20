import torch
import torch.nn as nn
from models.model_utils import AdaptiveEncoderCls_so

# Non-Parametric Network
class NPNet(nn.Module):
    def __init__(self, num_points=1024, init_dim=6, stages=4, stage_dim=72,
                k=90, sigma=0.3, feat_normalize=True,
    ):
        super().__init__()
        self.EncNP = AdaptiveEncoderCls_so(
            num_points, init_dim, stages, stage_dim, k, sigma, feat_normalize
        )

    def forward(self, xyz):     # [B, N, 3]
        x = self.EncNP(xyz)  # [B, embed_dim]
        return x


import time

if __name__ == "__main__":
    # Parameters
    batch_size = 128
    num_points = 1024
    in_ch = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_xyz = torch.randn(batch_size, num_points, in_ch).contiguous().to(device)
    model = NPNet(num_points=num_points).to(device)
    start_time = time.time()
    output = model(random_xyz)
    end_time = time.time()
    print(f"Output shape: {output.shape}")
    print(f"Time taken for forward pass: {end_time - start_time:.6f} seconds")
