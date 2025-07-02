# Non-Parametric Networks for 3D Point Cloud Part Segmentation
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

from .model_utils import *



# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long() 
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta) ################################################################################################

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Geometry Extraction
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x_w = self.geo_extract(knn_xyz, knn_x)

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_dim),
                nn.GELU())

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        lc_x = self.out_transform(lc_x)
        return lc_x



















import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NAPE(nn.Module):

    def __init__(self, in_dim, out_dim, sigma=0.26, baseline=0.1, scaling=10.0, eps=1e-6):    # baseline=0.1 > x
        super(NAPE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_sigma = sigma  # base kernel width
        self.baseline = baseline
        self.scaling = scaling
        self.eps = eps
        
        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()
        # Fixed grid of values for embedding (excluding endpoints)
        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)
        
    def forward(self, xyz): # b,in_d,n  8,3,1024
        """
        Args:
          xyz: Tensor of shape [B, N, in_dim] or [B, S, K, in_dim]
        Returns:
          Tensor of shape [B, ..., out_dim] computed by adaptively fusing a Gaussian 
          and a cosine response.
        """

        if xyz.shape[-1] != 3:
            xyz = xyz.permute(0,2,1)        # b,n,in_d  8,1024,3

        if self.out_dim == 0:
            return xyz
        if xyz.dim() == 3:
            # Compute global standard deviation across points (dim=1)
            global_std = torch.mean(torch.std(xyz, dim=1))
        elif xyz.dim() == 4:
            # Reshape to [B, -1, in_dim] and compute standard deviation over points
            global_std = torch.mean(torch.std(xyz.view(xyz.size(0), -1, self.in_dim), dim=1))
        else:
            raise ValueError("Input must be 3D or 4D")
        
        # Adaptive sigma: scale the base sigma by (1 + global_std)
        adaptive_sigma = self.base_sigma * (1 + global_std)
        # Adaptive blend weight via sigmoid; yields a value in (0,1)
        blend = torch.sigmoid((global_std - self.baseline) * self.scaling)
        
        embeds = []
        for i in range(self.in_dim):
            # Compute difference from fixed grid values
            tmp = xyz[..., i:i+1] - self.feat_val.to(xyz.device)
            # Gaussian (RBF) component using adaptive sigma
            rbf = (-0.5 * (tmp / (adaptive_sigma + self.eps))**2).exp()
            # Cosine component using the same adaptive sigma for scaling
            cosine = torch.cos(tmp / (adaptive_sigma + self.eps))
            # Adaptive fusion of the two components:
            combined = blend * rbf + (1 - blend) * cosine
            embeds.append(combined)
        
        # Concatenate all channels and select the desired output dimensions
        position_embed = torch.cat(embeds, dim=-1)
        position_embed = torch.index_select(position_embed, -1, self.out_idx.to(xyz.device))    # b,n,out_d  8,1024,64
        return position_embed     # .permute(0,2,1)     # b,n,out_d  8,1024,64




class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1000, beta=100, nape_ratio=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        
        # Allocate dimensions
        self.nape_dim = int(out_dim * nape_ratio)
        self.fourier_dim = out_dim - self.nape_dim

        # Modules
        self.nape = NAPE(in_dim, self.nape_dim)
    
    def forward(self, xyz):  # [B, 3, N]
        B, _, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1)  # [B, N, 3]

        # === Fourier Positional Encoding ===
        feat_dim = self.fourier_dim // (self.in_dim * 2)
        feat_range = torch.arange(feat_dim, device=xyz.device).float()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = self.beta * xyz.unsqueeze(-1) / dim_embed  # [B, 3, N, feat_dim]

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        fourier_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)  # [B, 3, N, 2*feat_dim]
        fourier_embed = fourier_embed.permute(0, 1, 3, 2).reshape(B, self.fourier_dim, N)

        # === NAPE Encoding ===
        nape_embed = self.nape(xyz_t)  # [B, N, nape_dim]
        nape_embed = nape_embed.permute(0, 2, 1)  # [B, nape_dim, N]

        # === Combine ===
        position_embed = torch.cat([fourier_embed, nape_embed], dim=1)  # [B, out_dim, N]
        return position_embed


class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1000, beta=100, nape_ratio=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

        # Allocate dimensions
        self.nape_dim = int(out_dim * nape_ratio)
        self.fourier_dim = out_dim - self.nape_dim

        # Modules
        self.nape = NAPE(in_dim, self.nape_dim)

    def forward(self, knn_xyz, knn_x):  # [B, 3, G, K], [B, out_dim, G, K]
        B, _, G, K = knn_xyz.shape
        xyz_t = knn_xyz.permute(0, 2, 3, 1)  # [B, G, K, 3]

        # === Fourier Positional Encoding ===
        feat_dim = self.fourier_dim // (self.in_dim * 2)
        feat_range = torch.arange(feat_dim, device=knn_xyz.device).float()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = self.beta * knn_xyz.unsqueeze(-1) / dim_embed  # [B, 3, G, K, feat_dim]

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        fourier_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)  # [B, 3, G, K, 2*feat_dim]
        fourier_embed = fourier_embed.permute(0, 1, 4, 2, 3).reshape(B, self.fourier_dim, G, K)

        # === NAPE Encoding ===
        nape_embed = self.nape(xyz_t)  # [B, G, K, nape_dim]
        nape_embed = nape_embed.permute(0, 3, 1, 2)  # [B, nape_dim, G, K]

        # === Combine ===
        position_embed = torch.cat([fourier_embed, nape_embed], dim=1)  # [B, out_dim, G, K]

        # === Weight knn_x ===
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w




'''
class PosE_Initial(nn.Module): # PosE_Initial_NAPE
    def __init__(self, in_dim, out_dim, sigma=0.4, baseline=0.1, scaling=10.0, eps=1e-6):
        super().__init__()
        self.nape = NAPE(in_dim, out_dim, sigma, baseline, scaling, eps)

    def forward(self, xyz):  # [B, 3, N]
        B, C, N = xyz.shape
        xyz = xyz.permute(0, 2, 1)  # -> [B, N, 3]
        pos_embed = self.nape(xyz)  # [B, N, out_dim]
        pos_embed = pos_embed.permute(0, 2, 1)  # -> [B, out_dim, N]
        return pos_embed


class PosE_Geo(nn.Module): # PosE_Geo_NAPE
    def __init__(self, in_dim, out_dim, sigma=0.4, baseline=0.1, scaling=10.0, eps=1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nape = NAPE(in_dim, out_dim, sigma, baseline, scaling, eps)

    def forward(self, knn_xyz, knn_x):  # [B, 3, G, K], [B, out_dim, G, K]
        B, C, G, K = knn_xyz.shape
        xyz_flat = knn_xyz.permute(0, 2, 3, 1)  # [B, G, K, 3]
        pos_embed = self.nape(xyz_flat)         # [B, G, K, out_dim]
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # -> [B, out_dim, G, K]

        knn_x_w = knn_x + pos_embed
        knn_x_w *= pos_embed
        return knn_x_w  # [B, out_dim, G, K]

'''















'''
# PosE for Raw-point Embedding 
class PosE_Initial(nn.Module):  ########################################################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim                # 3
        self.out_dim = out_dim              # 144
        self.alpha, self.beta = alpha, beta # 1000,100

    def forward(self, xyz):                 # 1,3,1024
        B, _, N = xyz.shape    
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim).float().cuda()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        
        return position_embed                # 1,144,1024


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):  ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim                    # 3   3   3    3
        self.out_dim = out_dim                  # 288 576 1152 2304
        self.alpha, self.beta = alpha, beta     # 1000 100
        
    def forward(self, knn_xyz, knn_x):          # [1,3,512,90 1,288,512,90] [1,3,256,90 1,576,256,90] [1,3,128,90 1,1152,128,90] [1,3,64,90 1,2304,64,90]
        print(knn_xyz.shape, knn_x.shape)
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w                      # 1,288,512,90   1,576,256,90   1,1152,128,90   1,2304,64,90
'''



































# Non-Parametric Encoder
class EncNP(nn.Module):  
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, alpha, beta):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(3, self.embed_dim, self.alpha, self.beta) #######################################################################

        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim))


    def forward(self, xyz, x):

        # Raw-point Embedding
        x = self.raw_point_embed(x)

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, C, N]

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)

            xyz_list.append(xyz)
            x_list.append(x)

        return xyz_list, x_list


# Non-Parametric Decoder
class DecNP(nn.Module):  
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
class Point_NN_Seg(nn.Module):
    def __init__(self, input_points=2048, num_stages=5, embed_dim=144, 
                    k_neighbors=128, de_neighbors=6, beta=1000, alpha=100):
        super().__init__()
        # Non-Parametric Encoder and Decoder
        self.EncNP = EncNP(input_points, num_stages, embed_dim, k_neighbors, alpha, beta)
        self.DecNP = DecNP(num_stages, de_neighbors)


    def forward(self, x):
        # xyz: point coordinates
        # x: point features
        xyz = x.permute(0, 2, 1)

        # Non-Parametric Encoder
        xyz_list, x_list = self.EncNP(xyz, x)
        # xyz:       1,1024,3
        #   x:       1,3,1024
        # xyz_list:  len: 5  : 1,1024,3   1,512,3   1,256,3   1,128,3    1,64,3
        # x_list:    len: 5  : 1,144,1024 1,288,512 1,576,256 1,1152,128 1,2304,64


        # Non-Parametric Decoder
        x = self.DecNP(xyz_list, x_list)    # x: 1,4464,1024
        return x