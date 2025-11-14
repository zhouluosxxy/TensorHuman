import torch
import torch.nn as nn
import tinycudann as tcnn
from torch.cuda.amp import autocast
import numpy as np
class CanonicalMLP(nn.Module):
    def __init__(self, **kwargs):
        super(CanonicalMLP, self).__init__()

        self.n_planes = 3
        self.n_feat = 8  # 每个平面的特征维度

        self.time_resolution = 540
        self.space_resolutions = [64, 128, 256, 512]# np.linspace(16, 512, 6).astype(int).tolist()#[16,32,64, 128, 256, 512]  # 空间多层次分辨率
        self.dim =  1 * self.n_feat * len(self.space_resolutions)
        self.plane_config = [(0, 1), (0, 2), (1, 2)]

        # 特征平面
        self.offset_feature = nn.ParameterList()
        for space_res in self.space_resolutions:
            for plane_type in range(self.n_planes):
                dim1, dim2 = self.plane_config[plane_type]
                res1 = space_res
                res2 = space_res
                plane = torch.randn(1, self.n_feat, res1, res2) * 0.2
                self.offset_feature.append(nn.Parameter(plane))

        # 更新 MLP 输入维度
        self.rgb_net = tcnn.Network(
            n_input_dims=self.dim ,  # 添加傅里叶编码后的特征维度
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        )
        self.sigma_net = tcnn.Network(
            n_input_dims=self.dim ,  # 添加傅里叶编码后的特征维度
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        )

    def normalize_coord(self, xyz_sampled, aabb_min=-2, aabb_max=2):
        aabb = torch.tensor([[aabb_min] * 3, [aabb_max] * 3], dtype=torch.float32).to(xyz_sampled.device)
        aabb_size = aabb[1] - aabb[0]
        return (xyz_sampled - aabb[0]) * (2.0 / aabb_size) - 1

    def sample_plane(self, plane, coord):
        coord = coord.unsqueeze(0).unsqueeze(0)
        features = torch.nn.functional.grid_sample(
            plane,
            coord,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        )
        return features.view(self.n_feat, -1).T

    def forward(self, pos_xyz, **kwargs):
        norm_xyz = self.normalize_coord(pos_xyz)
        xyz = norm_xyz

        features = []
        idx = 0

        # 使用共享的 time 进行多分辨率解码
        for space_res in self.space_resolutions:
            for plane_type in range(self.n_planes):
                dim1, dim2 = self.plane_config[plane_type]
                coord = xyz[..., [dim1, dim2]]
                plane = self.offset_feature[idx]
                feat = self.sample_plane(plane, coord)
                features.append(feat)
                idx += 1

        # 融合特征：乘积融合后直接拼接
        fused_features_list = []
        for i in range(len(self.space_resolutions)):
            start_idx = i * self.n_planes
            xy = features[start_idx + 0]
            xz = features[start_idx + 1]
            yz = features[start_idx + 2]
            fused_res = xy + xz + yz
            fused_features_list.append(fused_res)

        # 拼接多分辨率特征
        fused_features = torch.cat(fused_features_list, dim=-1)  # [N, 288]

        # 输入 MLP 解码
        rgb = self.rgb_net(fused_features)
        sigma = self.sigma_net(fused_features)
        rgb_sigma = torch.cat([rgb, sigma], dim=-1)

        return {"rgb_sigma": rgb_sigma}

