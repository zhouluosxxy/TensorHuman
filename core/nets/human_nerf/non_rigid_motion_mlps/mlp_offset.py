import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.network_util import initseq
import tinycudann as tcnn
import numpy as np


class NonRigidMotionMLP(nn.Module):
    def __init__(self,
                 pos_embed_size=36,
                 condition_code_size=69,
                 mlp_width=128,
                 mlp_depth=6,
                 skips=None,
                 n_voxel_init=16,
                 n_voxel_final=256,
                 upsamp_interval=5000,
                 n_iters=10000):
        super(NonRigidMotionMLP, self).__init__()

        self.n_planes = 6
        self.n_feat = 24
        self.plane_config = [(0, 1), (0, 3), (1, 3), (0, 2), (2, 3), (1, 2)]

        # 分辨率控制参数
        self.n_voxel_init = n_voxel_init
        self.n_voxel_final = n_voxel_final
        self.upsamp_interval = upsamp_interval
        self.n_iters = n_iters
        self.current_res = n_voxel_init

        # 仅维护当前分辨率的特征平面
        self.offset_feature = nn.ParameterList()
        for _ in range(self.n_planes):
            plane = torch.randn(1, self.n_feat, self.current_res, self.current_res) * 0.2
            self.offset_feature.append(nn.Parameter(plane))

        class SinActivation(nn.Module):
            def forward(self, x):
                return torch.sin(x)


        # 保持原始的trans_mlp不变
        self.trans_mlp = nn.Sequential(
            nn.Linear(24+69, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.time_mlp = tcnn.Network(
            n_input_dims=20,
            n_output_dims=1,
            network_config={
                "otype": "CutlassMLP",
                "activation": "Sine",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        )

        # 初始化
        #init_val = 1e-5
        #last_layer = self.trans_mlp[-1]
        #last_layer.weight.data.uniform_(-init_val, init_val)
        #last_layer.bias.data.zero_()

    def normalize_coord(self, xyz_sampled, aabb_min=-1.5, aabb_max=1.5):
        aabb = torch.tensor([[aabb_min] * 3, [aabb_max] * 3], dtype=torch.float32).to(xyz_sampled.device)
        aabb_size = aabb[1] - aabb[0]
        return (xyz_sampled - aabb[0]) * (2.0 / aabb_size) - 1

    def positional_encoding(self, x, L):
        freqs = 2.0 ** torch.linspace(0.0, L - 1, L, device=x.device)
        x_freqs = x[..., None] * freqs[None, None, :]
        enc = torch.cat([torch.sin(x_freqs), torch.cos(x_freqs)], dim=-1)
        return enc.view(x.shape[0], -1)

    def sample_plane(self, plane, coord):
        coord = coord.unsqueeze(0).unsqueeze(0)
        features = F.grid_sample(
            plane, coord,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        )
        return features.view(self.n_feat, -1).T

    def update_resolution(self, iter_val):
        """更新分辨率并正确使用new_offset_feature"""
        if self.n_voxel_final <= self.n_voxel_init:
            return self.current_res

        progress = min(iter_val / self.n_iters, 1.0)
        target_res = int(self.n_voxel_init * (self.n_voxel_final / self.n_voxel_init) ** progress)

        if target_res > self.current_res and iter_val % self.upsamp_interval == 0:
            # 创建新的特征平面列表
            new_offset_feature = nn.ParameterList()
            for plane in self.offset_feature:
                upsampled = F.interpolate(
                    plane.data,
                    size=(target_res, target_res),
                    mode='bilinear',
                    align_corners=True
                )
                new_offset_feature.append(nn.Parameter(upsampled))

            # 确保new_offset_feature被正确使用
            self.offset_feature = new_offset_feature
            self.current_res = target_res
            print(f"Upsampled to resolution: {self.current_res}")

        return self.current_res

    def forward(self, iter_val=None, pos_xyz=None, condition_code=None, time=None, **_):
        device = next(self.parameters()).device
        pos_xyz = pos_xyz.to(device)
        condition_code = condition_code.to(device)
        time = time.to(device)

        norm_xyz = self.normalize_coord(pos_xyz)
        time_fly = self.positional_encoding(time / 1000, 10)
        time = self.time_mlp(time_fly)

        # 更新分辨率
        self.update_resolution(iter_val if iter_val is not None else 0)

        # 多平面投影
        xyzt = torch.cat([norm_xyz, time], dim=-1)
        features = []

        for plane_type in range(self.n_planes):
            dim1, dim2 = self.plane_config[plane_type]
            coord = xyzt[..., [dim1, dim2]]
            plane = self.offset_feature[plane_type]
            feat = self.sample_plane(plane, coord)
            features.append(feat)

        # 特征乘积融合
        xy, xt, yt, xz, zt, yz = features
        fused_xy_zt = xy * zt
        fused_xz_yt = xz * yt
        fused_yz_xt = yz * xt

        # 特征拼接
        fused_features = fused_xy_zt+ fused_xz_yt+ fused_yz_xt# torch.cat([fused_xy_zt, fused_xz_yt, fused_yz_xt], dim=-1)#fused_xy_zt+ fused_xz_yt+ fused_yz_xt#
        fused_features = torch.cat([fused_features, condition_code], dim=-1)

        trans = self.trans_mlp(fused_features)
        trans = torch.tanh(trans)

        return {
            'xyz': pos_xyz + trans*0.1,
            'offset': trans
        }
