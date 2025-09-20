import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

project_path = os.path.abspath(".")
print(project_path)
sys.path.append(project_path)

from models.model_utils import *


# Non-Parametric Decoder
class Decoder(nn.Module):  
    def __init__(self, num_stages, de_neighbors):
        super().__init__()
        self.num_stages = num_stages
        self.de_neighbors = de_neighbors


    def propagate(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            weight = weight.view(B, N, self.de_neighbors, 1)

            index_points(xyz1, idx)
            interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)

        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        return new_points


    def forward(self, xyz_list, x_list):
        xyz_list.reverse()
        x_list.reverse()

        x = x_list[0]
        for i in range(self.num_stages):
            # Propagate point features to neighbors
            x = self.propagate(xyz_list[i+1], xyz_list[i], x_list[i+1], x)
        return x


# Non-Parametric Network
class NPNet_Seg(nn.Module):
    def __init__(self, input_points=2048, num_stages=5, embed_dim=144, 
                    k_neighbors=128, de_neighbors=6, beta=1000, alpha=100,
                    sigma=0.26, baseline=0.1, scaling=10.0, eps=1e-6, encoder_type='seg', adaptive_ratio=0.5):
        super().__init__()
        # Non-Parametric Encoder and Decoder
        if encoder_type == 'seg':
            self.AdaptiveEncoderSeg = AdaptiveEncoderSeg(input_points, num_stages, embed_dim, k_neighbors, sigma, baseline, scaling, eps, alpha, beta, adaptive_ratio)
        else:
            from models.npnet_cls_mn40 import AdaptiveEncoderCls
            self.AdaptiveEncoderSeg = AdaptiveEncoderCls(num_points=input_points, init_dim=embed_dim, stages=num_stages, stage_dim=embed_dim,
                                                         k=k_neighbors, sigma=sigma, baseline=baseline, scaling=scaling, eps=eps, feat_normalize=True)

        self.Decoder = Decoder(num_stages, de_neighbors)


    def forward(self, x):           # B,N,3  64,3,1024
        # xyz: point coordinates
        # x: point features
        xyz = x.permute(0, 2, 1)    # B,N,3  64,1024,3

        # Non-Parametric Encoder
        xyz_list, x_list = self.AdaptiveEncoderSeg(xyz, x)  # 
        # xyz:       1,1024,3
        #   x:       1,3,1024
        # xyz_list:  len: 5  : 1,1024,3   1,512,3   1,256,3   1,128,3    1,64,3
        # x_list:    len: 5  : 1,144,1024 1,288,512 1,576,256 1,1152,128 1,2304,64
        # test
        # xyz_list: 2,2048,3  2,1024,3  2,512,3  2,256,3  2,128,3  2,64,3
        # x_list  : 2,144,2048  2,288,1024  2,756,512  2,1152,256  2,2304,128  2,4608,64

        # Non-Parametric Decoder
        x = self.Decoder(xyz_list, x_list)    # x: 1,4464,1024
        return x
    

if __name__ == "__main__":
    import time
    batch_size = 2
    num_points = 2048
    in_ch = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    random_xyz = torch.randn(batch_size, in_ch, num_points).contiguous().to(device)

    model = NPNet_Seg(input_points=num_points, encoder_type='seg').to(device)
    start_time = time.time()
    output = model(random_xyz)
    end_time = time.time()
    print(f"Output shape: {output.shape}")
    print(f"Time taken for forward pass: {end_time - start_time:.6f} seconds")
