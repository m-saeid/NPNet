import os
import sys
import math
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

project_path = os.path.abspath(".")
sys.path.append(project_path)

from pointnet2_ops import pointnet2_utils
# from pytorch3d.ops import sample_farthest_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def normalize_tensor(data_tensor, knn_tensor, with_center=True):
    if with_center:
        center_tensor = data_tensor.unsqueeze(dim=-2)
        knn_std = torch.std(knn_tensor - center_tensor, dim=(0, 1, 3), keepdim=True).clamp(min=1e-5)
        normalized_knn_tensor = (knn_tensor - center_tensor) / knn_std
    else:
        knn_std = torch.std(knn_tensor, dim=(0, 1), keepdim=True).clamp(min=1e-5)
        normalized_knn_tensor = knn_tensor / knn_std
    return normalized_knn_tensor


class AdaptiveEmbedding(nn.Module):
    """
    Adaptive Embedding Function (AdaptiveEmbedding)
    
    This function implements an adaptive, data-driven variant of the standard
    Gaussian (RBF) embedding. It adjusts the kernel width (sigma) based on the 
    global standard deviation of the input and uses an adaptive blending strategy
    to fuse the Gaussian response with a complementary cosine response.
    
    Assumptions and Implementation Details:
    
    1. Adaptive Kernel Width:
       - Compute a global standard deviation from the input (over points) and 
         adjust the effective sigma as: adaptive_sigma = base_sigma * (1 + global_std).
         
    2. Adaptive Blending:
       - Compute a blend weight (between 0 and 1) as: 
           blend = sigmoid((global_std - baseline) * scaling)
         This weight is used to fuse the Gaussian (RBF) embedding with a cosine embedding.
         
    3. Dynamic Normalization:
       - The difference (tmp) is divided by the adaptive sigma to normalize the scale 
         of the kernel function.
    
    4. Complementarity:
       - The Gaussian captures local similarity via an exponential decay,
         whereas the cosine transformation introduces a periodic component.
         Their fusion is intended to yield a richer representation.
    
    5. Parameterlessness:
       - All adaptation is computed on-the-fly from the data, with no learnable parameters.
    
    Args:
      in_dim (int): Input dimension (typically 3 for XYZ coordinates).
      out_dim (int): Desired output dimension.
      sigma (float): Base sigma value (default kernel width).
      baseline (float): A fixed baseline for computing blend weight (default 0.1).
      scaling (float): Scaling factor for the sigmoid to compute blend (default 10.0).
      eps (float): Small constant to prevent division by zero.
    """
    def __init__(self, in_dim, out_dim, sigma=0.26, baseline=0.1, scaling=10.0, eps=1e-6, fixed_sigma=None, fixed_blend=None):
        super(AdaptiveEmbedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_sigma = sigma  # base kernel width
        self.baseline = baseline
        self.scaling = scaling
        self.eps = eps
        self.fixed_sigma = fixed_sigma
        self.fixed_blend = fixed_blend
        
        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()
        # Fixed grid of values for embedding (excluding endpoints)
        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)
        
    def forward(self, xyz):
        """
        Args:
          xyz: Tensor of shape [B, N, in_dim] or [B, S, K, in_dim]
        Returns:
          Tensor of shape [B, ..., out_dim] computed by adaptively fusing a Gaussian 
          and a cosine response.
        """

        #if xyz.shape[-1] != 3:
        #    xyz = xyz.permute(0,2,1)

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
        adaptive_sigma = self.base_sigma * (1 + global_std) if self.fixed_sigma is None else self.fixed_sigma
        # Adaptive blend weight via sigmoid; yields a value in (0,1)
        blend = torch.sigmoid((global_std - self.baseline) * self.scaling) if self.fixed_blend is None else self.fixed_blend
        
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
        position_embed = torch.index_select(position_embed, -1, self.out_idx.to(xyz.device))
        return position_embed


class Grouper(nn.Module):
    def __init__(self, stage_points, k, feat_normalize):
        super().__init__()
        self.stage_points = stage_points
        self.k = k
        self.feat_normalize = feat_normalize

    def forward(self, xyz, feat):
        b, n, _ = xyz.shape
        # if xyz.device == torch.device("cpu"):
        #     fps_idx = farthest_point_sample(xyz.contiguous(), self.stage_points).long()
        # else:
        #     fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), self.stage_points).long()
        #     # fps_idx = farthest_point_sample(xyz.contiguous(), self.stage_points).long()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), self.stage_points).long()
        xyz_sampled = index_points(xyz, fps_idx)
        feat_sampled = index_points(feat, fps_idx)
        idx_knn = knn_point(self.k, xyz, xyz_sampled)
        xyz_knn = index_points(xyz, idx_knn)
        feat_knn = index_points(feat, idx_knn)
        xyz_knn = normalize_tensor(xyz_sampled, xyz_knn, with_center=True)
        feat_knn = normalize_tensor(feat_sampled, feat_knn, with_center=self.feat_normalize)
        b, s, k, _ = feat_knn.shape
        feat_knn = torch.cat([feat_knn, feat_sampled.reshape(b, s, 1, -1).repeat(1, 1, k, 1)], dim=-1)
        return xyz_sampled, feat_sampled, xyz_knn, feat_knn


class Aggregation(nn.Module):
    def __init__(self, out_dim, sigma, baseline, scaling, eps, adaptive_embedding, fixed_sigma, fixed_blend):
        super().__init__()
        self.adaptive_embed = adaptive_embedding(3, out_dim, sigma, baseline, scaling, eps, fixed_sigma, fixed_blend)

    def forward(self, xyz_knn, feat_knn):
        # If the embedding function requires neighbor_xyz (i.e. if its forward method takes 2 arguments),
        # pass xyz_knn as both the input and the neighbor information.
        position_embed = self.adaptive_embed(xyz_knn)
        feat_knn_w = feat_knn + position_embed
        feat_knn_w *= position_embed
        return feat_knn_w


class Aggregation_so(nn.Module):
    def __init__(self, out_dim, sigma):
        super().__init__()
        self.gpe_embed = AdaptiveEmbedding(in_dim=3,
                                            out_dim=out_dim,
                                            sigma=sigma)

    def forward(self, xyz_knn, feat_knn):
        position_embed = self.gpe_embed(xyz_knn)  # [B, S, K, out_dim]
        feat_knn_w = feat_knn + position_embed  # [B, S, K, out_dim]
        feat_knn_w *= position_embed  # [B, S, K, out_dim]

        return feat_knn_w  # [B, S, K, out_dim]


class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(nn.GELU())

    def forward(self, feat_knn_w):
        feat_agg = feat_knn_w.mean(-2) + feat_knn_w.max(-2)[0]      # [B, S, D]
        feat_agg = self.out_transform(feat_agg.transpose(-2, -1))   # [B, D, S]
        return feat_agg.transpose(-2, -1)                           # [B, S, D]


class AdaptiveEncoderCls(nn.Module):
    def __init__(self, num_points, init_dim, stages, stage_dim, k, sigma, baseline, scaling, eps, feat_normalize, fixed_sigma, fixed_blend, **kwargs):
        super().__init__()
        self.num_points = num_points
        self.init_dim = init_dim
        self.stages = stages
        self.stage_dim = stage_dim
        self.sigma = sigma
        self.k = k
        # self.fixed_sigma = fixed_sigma
        # self.fixed_blend = fixed_blend

        # Initialize embedding function with the correct arguments
        self.init_embed = AdaptiveEmbedding(in_dim=3,
                                            out_dim=init_dim,
                                            sigma=sigma,
                                            baseline=baseline, scaling=scaling, eps=eps,
                                            fixed_sigma=fixed_sigma, fixed_blend=fixed_blend)
                                            # baseline=0.1, scaling=10.0, eps=1e-6

        self.grp_list = nn.ModuleList()
        self.agg_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()

        out_dim = self.init_dim if self.init_dim != 0 else 3
        stage_points = self.num_points

        for i in range(self.stages):
            out_dim *= 2
            stage_points //= 2
            self.grp_list.append(Grouper(stage_points, k, feat_normalize))
            self.agg_list.append(Aggregation(out_dim, self.sigma, baseline, scaling, eps,
                                             adaptive_embedding=AdaptiveEmbedding, fixed_sigma=fixed_sigma, fixed_blend=fixed_blend))  
            self.pool_list.append(Pooling(out_dim))


    def forward(self, xyz, x=None):             # B,N,3  64,1024,3    x: B,3,N  64,3,1024
        if x is not None:               # Segmentation
            feat = self.init_embed(x.permute(0,2,1))
            xyz_list = [xyz]
            x_list = [feat.permute(0,2,1)]
        else:                           # Classification
            feat = self.init_embed(xyz)     # B,N,C - 64,1024,6

        stage_results = []
        for i in range(self.stages):
            xyz, feat, xyz_knn, feat_knn = self.grp_list[i](xyz, feat)   # 64,512,3 - 64,512,6 - 64,512,90,3 - 64,512,90,12
            feat_knn_w = self.agg_list[i](xyz_knn, feat_knn)
            feat = self.pool_list[i](feat_knn_w)

            if x is None:   # Classification
                stage_pooling = torch.cat((feat.max(-2)[0], feat.mean(-2)), dim=1)
                stage_results.append(stage_pooling)
            else:           # Segmentation
                xyz_list.append(xyz)
                x_list.append(feat.permute(0,2,1))

        if x is None:   # Classification
            encoded_out = torch.cat(stage_results, dim=1)
            return encoded_out
        else:           # Segmentation
            return xyz_list, x_list


#####################################
############### SEG #################
#####################################


class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, sigma, baseline, scaling, eps, alpha, beta, adaptive_ratio):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

        # Allocate dimensions
        self.adaptive_dim = int(out_dim * adaptive_ratio)
        self.fourier_dim = out_dim - self.adaptive_dim

        # Modules
        self.adaptiveembedding = AdaptiveEmbedding(in_dim, self.adaptive_dim, sigma, baseline, scaling, eps)

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

        # === Adaptive Encoding ===
        adaptive_embed = self.adaptiveembedding(xyz_t)  # [B, G, K, adaptive_dim]
        adaptive_embed = adaptive_embed.permute(0, 3, 1, 2)  # [B, adaptive_dim, G, K]

        # === Combine ===
        position_embed = torch.cat([fourier_embed, adaptive_embed], dim=1)  # [B, out_dim, G, K]

        # === Weight knn_x ===
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w


class Grouper_seg(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        # IF : pointnet2_ops import pointnet2_utils
        # try:
        #     fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long()
        # except:
        #     fps_idx = farthest_point_sample(xyz, self.group_num).long()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long()
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)
        '''
        # ELIF : pytorch3d.ops import sample_farthest_points
        lc_xyz, fps_idx = sample_farthest_points(xyz, K=self.group_num) #.long()
        lc_x = index_points(x, fps_idx)
        '''

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x
    

# Local Geometry Aggregation
class Aggregation_seg(nn.Module):
    def __init__(self, out_dim, sigma=0.26, baseline=0.1, scaling=10.0, eps=1e-6, alpha=1000, beta=100, adaptive_ratio=0.5):
        super().__init__()
        self.geo_extract = PosE_Geo(3, out_dim, sigma, baseline, scaling, eps, alpha, beta, adaptive_ratio)

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
class Pooling_seg(nn.Module):
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
    


class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, sigma, baseline, scaling, eps, alpha, beta, adaptive_ratio):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        
        # Allocate dimensions
        self.adaptive_dim = int(out_dim * adaptive_ratio)
        self.fourier_dim = out_dim - self.adaptive_dim

        # Modules
        self.adaptiveembedding = AdaptiveEmbedding(in_dim, self.adaptive_dim, sigma, baseline, scaling, eps)
    
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

        # === Adaptive Encoding ===
        adaptive_embed = self.adaptiveembedding(xyz_t)  # [B, N, adaptive_dim]
        adaptive_embed = adaptive_embed.permute(0, 2, 1)  # [B, adaptive_dim, N]

        # === Combine ===
        position_embed = torch.cat([fourier_embed, adaptive_embed], dim=1)  # [B, out_dim, N]
        return position_embed


# Non-Parametric Encoder
class AdaptiveEncoderSeg(nn.Module):  
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, sigma=0.26, baseline=0.1, scaling=10.0, eps=1e-6, alpha=1000, beta=100, adaptive_ratio=0.5):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(3, self.embed_dim, sigma, baseline, scaling, eps, alpha, beta, adaptive_ratio)

        self.grp_list = nn.ModuleList() # FPS, kNN
        self.agg_list = nn.ModuleList() # Local Geometry Aggregation
        self.pool_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            group_num = group_num // 2
            self.grp_list.append(Grouper_seg(group_num, k_neighbors))
            self.agg_list.append(Aggregation_seg(out_dim, sigma, baseline, scaling, eps, alpha, beta, adaptive_ratio))
            self.pool_list.append(Pooling_seg(out_dim))

    def forward(self, xyz, x):          # xyz: B,N,3  64,1024,3     x: B,3,N  64,3,1024

        # Raw-point Embedding
        x = self.raw_point_embed(x)     # x: B,3,N  64,3,1024  >  x: B,C,N  64,144,1024

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]      # [B, C, N]

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.grp_list[i](xyz, x.permute(0, 2, 1))   # 64,512,3 - 64,512,144 - 64,512,128,3 - 64,512,128,144
            # Local Geometry Aggregation
            knn_x_w = self.agg_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.pool_list[i](knn_x_w)

            xyz_list.append(xyz)
            x_list.append(x)

        return xyz_list, x_list
    

###################################
########## ScanObject #############
###################################


def process_data(data_loader, model, device):
    features_list, labels_list = [], []

    # Loop through the provided data loader
    for points, labels in tqdm(data_loader, leave=False):
        # points: [B, num_points, 3]
        point_features = model(points.to(device))  # [B, num_features]
        features_list.append(point_features)

        labels = labels.to(device)
        labels_list.append(labels)

    features = torch.cat(features_list, dim=0)  # [num_samples, num_features]
    features = F.normalize(features, dim=-1)
    # features = features.permute(1, 0)  # [num_features, num_samples]

    labels = torch.cat(labels_list, dim=0)  # [num_samples, 1]

    return features, labels
try:
    import general_utils_so as gutils
except:
    pass

def process_and_evaluate(train_loader, test_loader, model, device):
    # Process training data
    start_train_time = time.time()
    train_features, train_labels = process_data(train_loader, model, device)
    train_labels = F.one_hot(train_labels).squeeze().float()
    train_time = time.time() - start_train_time

    # Process testing data
    start_test_time = time.time()
    test_features, test_labels = process_data(test_loader, model, device)
    test_time = time.time() - start_test_time

    # Calculate accuracies
    acc_cos, gamma = gutils.cosine_similarity(
        test_features, train_features, train_labels, test_labels
    )
    acc_1nn = gutils.one_nn_classification(
        test_features, train_features, train_labels, test_labels
    )

    # Return results
    return {
        "train_time": train_time,
        "test_time": test_time,
        "acc_1nn": acc_1nn,
        "acc_cos": acc_cos,
        "gamma": gamma,
    }



class AdaptiveEncoderCls_so(nn.Module):
    def __init__(
        self, num_points, init_dim, stages, stage_dim, k, sigma, feat_normalize
    ):
        super().__init__()
        self.num_points = num_points
        self.init_dim = init_dim
        self.stages = stages
        self.stage_dim = stage_dim
        self.sigma = sigma

        # Initial Embedding
        # self.init_embed = EmbeddingGPE(3, self.init_dim, sigma)
        self.init_embed = AdaptiveEmbedding(in_dim=3,
                                            out_dim=init_dim,
                                            sigma=sigma)

        self.lg_list = nn.ModuleList()  # FPS, kNN
        self.agpe_list = nn.ModuleList()  # GPE Aggregation
        self.pool_list = nn.ModuleList()  # Pooling

        out_dim = self.init_dim if self.init_dim != 0 else 3
        stage_points = self.num_points

        # Multi-stage Hierarchy
        for i in range(self.stages):
            out_dim = out_dim * 2
            stage_points = stage_points // 2
            self.lg_list.append(Grouper(stage_points, k, feat_normalize))
            self.agpe_list.append(Aggregation_so(out_dim, self.sigma))
            self.pool_list.append(Pooling(out_dim))

    def forward(self, xyz):
        # xyz: point coordinates    # [B, N, 3]

        # Initial Embedding
        feat = self.init_embed(xyz)  # [B, N, init_dim]

        stage_results = []
        skip_feat = feat  # For skip connections

        # Multi-stage Hierarchy
        for i in range(self.stages):
            # FPS, kNN
            xyz, feat, xyz_knn, feat_knn = self.lg_list[i](xyz, feat)

            # GPE Aggregation
            feat_knn_w = self.agpe_list[i](xyz_knn, feat_knn)
            # [B, N/2^i, K, D_i * 2]

            # Neighbor Pooling
            feat = self.pool_list[i](feat_knn_w)  # [B, N/2^i, D_i * 2]

            # Stage Pooling
            stage_pooling = torch.cat((feat.max(-2)[0], feat.mean(-2)), dim=1)
            # [B, D_i * 4]

            stage_results.append(stage_pooling)

        # encoded_out = feat.max(-2)[0] + feat.mean(-2)  # [B, dim = 96]
        encoded_out = torch.cat(stage_results, dim=1)  # [B, embed_dim)
        return encoded_out