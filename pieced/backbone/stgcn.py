import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union

from pieced.backbone.utils.tgcn import ConvTemporalGraphical
from pieced.backbone.utils.graph import Graph

KINECT_V2_PARTS = {
    'part_names': ['trunk', 'l_leg', 'r_leg', 'l_arm', 'r_arm'],
    'joint_indices': [
        [0, 1, 2, 3, 4, 8, 12, 16, 20],  # trunk
        [12, 13, 14, 15],                # l_leg
        [16, 17, 18, 19],                # r_leg
        [4, 5, 6, 7, 21, 22],            # l_arm
        [8, 9, 10, 11, 23, 24]           # r_arm
    ]
}

# [수정] OpenPose 18 Layout (for Kinetics-400)
KINETICS_PARTS = {
    'part_names': ['trunk', 'l_leg', 'r_leg', 'l_arm', 'r_arm'],
    'joint_indices': [
        # Trunk: Nose(0), Neck(1), R/L-Shoulder(2,5), R/L-Hip(8,11), Eyes(14,15), Ears(16,17)
        [0, 1, 2, 5, 8, 11, 14, 15, 16, 17], 
        # L_Leg: L-Hip(11), L-Knee(12), L-Ankle(13)
        [11, 12, 13], 
        # R_Leg: R-Hip(8), R-Knee(9), R-Ankle(10)
        [8, 9, 10], 
        # L_Arm: L-Shoulder(5), L-Elbow(6), L-Wrist(7)
        [5, 6, 7], 
        # R_Arm: R-Shoulder(2), R-Elbow(3), R-Wrist(4)
        [2, 3, 4] 
    ]
}

# [수정] Dataset Name & Layout to Part Mapping
PART_MAPPINGS = {
    'ntu-rgb+d': KINECT_V2_PARTS,
    'ntu-rgb+d-120': KINECT_V2_PARTS,
    'ntu60': KINECT_V2_PARTS,
    'ntu120': KINECT_V2_PARTS,
    'pkuv1': KINECT_V2_PARTS,
    'pkuv2': KINECT_V2_PARTS,
    
    # Kinetics / OpenPose 지원 추가
    'kinetics400': KINETICS_PARTS,
    'openpose': KINETICS_PARTS       # layout='openpose'로 들어올 경우 대응
}


class PartLevelPooling(nn.Module):
    # [수정] pool_type 파라미터 추가 (기본값 'max')
    def __init__(self, graph_layout: str, pool_type: str = 'max'):
        super().__init__()
        
        # graph_layout 이름이 매핑에 없으면 에러
        if graph_layout not in PART_MAPPINGS:
            raise ValueError(f"Body part mapping is not defined for graph layout '{graph_layout}'. defined: {list(PART_MAPPINGS.keys())}")

        valid_pool_types = ['max', 'avg']
        if pool_type not in valid_pool_types:
            raise ValueError(f"pool_type must be one of {valid_pool_types}. (Received: {pool_type})")

        self.pool_type = pool_type
        self.part_names = PART_MAPPINGS[graph_layout]['part_names']
        joint_indices_list = PART_MAPPINGS[graph_layout]['joint_indices']
        
        for name, indices in zip(self.part_names, joint_indices_list):
            self.register_buffer(f"indices_{name}", torch.tensor(indices, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        part_vectors = []
        for name in self.part_names:
            part_indices = getattr(self, f"indices_{name}")
            x_part = x.index_select(dim=3, index=part_indices)
            
            # [수정] pool_type에 따른 분기 처리
            if self.pool_type == 'max':
                z_part = F.max_pool2d(x_part, x_part.size()[2:])
            elif self.pool_type == 'avg':
                z_part = F.avg_pool2d(x_part, x_part.size()[2:])
                
            part_vectors.append(z_part.squeeze(-1).squeeze(-1))
        return part_vectors


class STGCN(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    # [수정] pool_type 파라미터 추가
    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, dropout, pooling_mode: str = 'whole',
                 pool_type: str = 'max'):
        super().__init__()

        # Store pooling options
        valid_modes = ['whole', 'part']
        if pooling_mode not in valid_modes:
            raise ValueError(f"pooling_mode must be one of {valid_modes}. (Received: {pooling_mode})")
        self.pooling_mode = pooling_mode
        self.pool_type = pool_type
        self.graph_layout = graph_args.get('layout', 'ntu-rgb+d')

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, dropout, residual=False),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, dropout),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, dropout),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, dropout),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, dropout),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, dropout),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, dropout),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, dropout),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, dropout),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, dropout),
        ))
        
        # [수정] PartLevelPooling 초기화 시 pool_type 전달
        if self.pooling_mode != 'whole':
            self.part_pool = PartLevelPooling(self.graph_layout, pool_type=self.pool_type)

        # self.fc is required by base.py to reference features_dim.
        # It will be overwritten by nn.Identity() in base.py.
        self.fc = nn.Linear(hidden_dim, num_class)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        
        # Current shape of x: (N*M, hidden_dim, T_out, V)

        # --- Pooling Branch ---
        
        # 1. Calculate Global (Whole) Pooling
        if self.pooling_mode in ['whole']:
            # [수정] Global pooling에서도 동일하게 pool_type 분기 적용
            if self.pool_type == 'max':
                x_global_pooled = F.max_pool2d(x, x.size()[2:])
            elif self.pool_type == 'avg':
                x_global_pooled = F.avg_pool2d(x, x.size()[2:])
            
            # (N*M, hidden_dim)
            x_global_flat = x_global_pooled.squeeze(-1).squeeze(-1)
            # (N, hidden_dim) - average over M dimension
            x_global = x_global_flat.view(N, M, -1).mean(dim=1)
        
        # 2. Calculate Part Pooling
        if self.pooling_mode in ['part']:
            # Returns a list of 5 tensors, each (N*M, hidden_dim)
            x_parts_pooled_list = self.part_pool(x)
            
            # Average each tensor over the M dimension
            x_parts_list = [
                part_tensor.view(N, M, -1).mean(dim=1) 
                for part_tensor in x_parts_pooled_list
            ]

        # --- Determine Return Value ---
        if self.pooling_mode == 'whole':
            return x_global  # Tensor (N, hidden_dim)

        elif self.pooling_mode == 'part':
            return x_parts_list  # List of 5 Tensors, each (N, hidden_dim)
        
        else:
            # This case was already checked in __init__, but added for safety
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A