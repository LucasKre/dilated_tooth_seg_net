import numpy as np
import torch
from torch import nn
try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps
except Exception as e:
    print("Warning: Cannot import furthest point sample. Using a slower version. Did install the PointNet++ Ops Lib? (See README.md)")
    def fps(xyz, npoint):
        """
        Farthest Point Sampling (FPS) algorithm for selecting a subset of points from a point cloud.

        Args:
            xyz (torch.Tensor): Input point cloud tensor of shape (B, N, C), where B is the batch size, N is the number of points, and C is the number of dimensions.
            npoint (int): Number of points to select.

        Returns:
            torch.Tensor: Tensor of shape (B, npoint) containing the indices of the selected points.
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
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids


def knn(x, k=16):
    """
    Performs k-nearest neighbors (knn) search on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_points, num_dims).
        k (int): Number of nearest neighbors to find.

    Returns:
        torch.Tensor: Index tensor of shape (batch_size, num_points, k), containing the indices of the k nearest neighbors for each point.
    """
    x_t = x.transpose(2, 1)
    pairwise_distance = torch.cdist(x_t, x_t, p=2)
    idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx



def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def get_graph_feature(x, k=20, idx=None, pos=None, edge_function='global'):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if pos is None:
            idx = knn(x, k=k)
        else:
            idx = knn(pos, k=k)
    device = x.device

    idx_org = idx

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if edge_function == 'global':
        feature = feature.permute(0, 3, 1, 2).contiguous()
    elif edge_function == 'local':
        feature = (feature - x).permute(0, 3, 1, 2).contiguous()
    elif edge_function == 'local_global':
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature, idx_org  # (batch_size, 2*num_dims, num_points, k)




class EdgeGraphConvBlock(nn.Module):
    """
    EdgeGraphConvBlock is a module that performs edge graph convolution on input features.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        edge_function (str): Type of edge function to use. Can be "global", "local", or "local_global".
        k (int): Number of nearest neighbors to consider for local edge function. Default is 32.

    Raises:
        ValueError: If edge_function is not one of "global", "local", or "local_global".

    Attributes:
        edge_function (str): Type of edge function used.
        in_channels (int): Number of input channels.
        k (int): Number of nearest neighbors considered for local edge function.
        conv (nn.Sequential): Sequential module consisting of convolutional layers.

    """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_function, k=32):
        super(EdgeGraphConvBlock, self).__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        self.k = k
        if edge_function not in ["global", "local", "local_global"]:
            raise ValueError(
                f'Edge Function {edge_function} is not allowed. Only "global", "local" or "local_global" are valid')
        if edge_function == "local_global":
            self.in_channels = self.in_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, pos=None, idx=None):
        """
        Forward pass of the EdgeGraphConvBlock.

        Args:
            x (torch.Tensor): Input features.
            idx (torch.Tensor, optional): Index tensor for graph construction of shape (B, N, K), where B is the batch size. Defaults to None.
            pos (torch.Tensor, optional): Position tensor of shape (B, N, D), where D is the number of dimensions. Default is None.

        Returns:
            torch.Tensor: Output features after edge graph convolution.
            torch.Tensor: Updated index tensor.

        """
        x_t = x.transpose(2, 1)
        if pos is None:
            pos = x
        pos_t = pos.transpose(2, 1)
        out, idx = get_graph_feature(x_t, edge_function=self.edge_function, k=self.k, idx=idx, pos=pos_t)
        out = self.conv(out)
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.transpose(2, 1)
        return out, idx


class DilatedEdgeGraphConvBlock(nn.Module):
    """
    A block implementing a dilated edge graph convolution operation.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        edge_function (str): Type of edge function to use. Must be one of "global", "local", or "local_global".
        dilation_k (int): Number of nearest neighbors to consider for the dilation operation.
        k (int): Number of nearest neighbors to consider for the graph convolution operation.

    Raises:
        ValueError: If `dilation_k` is smaller than `k` or if `edge_function` is not one of the allowed values.

    """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_function, dilation_k=128, k=32):
        super(DilatedEdgeGraphConvBlock, self).__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        if dilation_k < k:
            raise ValueError(f'Dilation k {dilation_k} must be larger than k {k}')
        self.dilation_k = dilation_k
        self.k = k
        if edge_function not in ["global", "local", "local_global"]:
            raise ValueError(
                f'Edge Function {edge_function} is not allowed. Only "global", "local" or "local_global" are valid')
        if edge_function in ["local_global"]:
            self.in_channels = self.in_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, pos, cd=None):
        """
        Forward pass of the dilated edge graph convolution block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size, N is the number of nodes,
                and C is the number of input channels.
            pos (torch.Tensor, optional): Position tensor of shape (B, N, D), where D is the number of dimensions.
                Defaults to None.
            cd (torch.Tensor, optional): Pairwise distance tensor of shape (B, N, N). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, N, out_channels), where out_channels is the number of output channels.
            torch.Tensor: Index tensor of shape (B, N, K), representing the indices of the nearest neighbors.

        """
        x_t = x.transpose(2, 1)
        B, N, C = x.shape
        if cd is None:
            cd = torch.cdist(pos, pos, p=2)
        dilation_k = min(self.dilation_k, N)
        idx_l = torch.topk(cd, dilation_k, largest=False)[1].reshape(B * N, -1)
        idx_fps = fps(pos.reshape(B * N, -1)[idx_l], self.k).long()
        idx_fps = batched_index_select(idx_l, 1, idx_fps).reshape(B, N, -1)
        out, idx = get_graph_feature(x_t, edge_function=self.edge_function, k=self.k, idx=idx_fps)
        out = self.conv(out)
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.transpose(2, 1)
        return out, idx


class GraphGroupSelfAttention(nn.Module):
    """
    Graph Group Self-Attention module.

    Args:
        in_channels (int): Number of input channels.
        group_k (int, optional): Number of groups to divide the input into. Default is 32.
        num_heads (int, optional): Number of attention heads. Default is 3.
        dropout (float, optional): Dropout probability. Default is 0.1.
    """

    def __init__(self, in_channels, group_k=32, num_heads=3, dropout=0.1):
        super(GraphGroupSelfAttention, self).__init__()
        self.group_k = group_k
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, pos):
        """
        Forward pass of the GraphGroupSelfAttention module.
        :param x: Input tensor of shape (B, N, C), where B is the batch size, N is the number of nodes, and C is the number of input channels.
        :return: Output tensor of shape (B, N, C), representing the output of the GraphGroupSelfAttention module.
        """
        group_idx = fps(pos, self.group_k)
        groups = batched_index_select(x, 1, group_idx)  # (B, N, C) -> (B, group_k, C)
        attn_output, attn_output_weights = self.multihead_attn(x, groups, groups)
        out = attn_output + x
        return out


class BasicPointLayer(nn.Module):
    """
    Basic point layer consisting of a 1D convolution, batch normalization, leaky ReLU, and dropout.
    """

    def __init__(self, in_channels, out_channels, dropout=0.1, is_out=False):
        """
        Initializes the BasicPointLayer.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param dropout: Dropout probability. Default is 0.1.
        """
        super(BasicPointLayer, self).__init__()
        if is_out:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        x = x.transpose(2, 1)
        return self.conv(x).transpose(2, 1)



class ResidualBasicPointLayer(nn.Module):
    """
    Basic point layer consisting of a 1D convolution, batch normalization, leaky ReLU, and dropout
    with a residual connection.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        """
        Initializes the BasicPointLayer.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param dropout: Dropout probability. Default is 0.1.
        """
        super(ResidualBasicPointLayer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)
        )
        if in_channels != out_channels:
            self.rescale = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout)
            )
        else:
            self.rescale = nn.Identity()

    def forward(self, x):
        x = x.transpose(2, 1)
        return self.conv(x).transpose(2, 1) + self.rescale(x).transpose(2, 1)


class PointFeatureImportance(nn.Module):
    """
    Point Feature Importance module.
    """

    def __init__(self, in_channels):
        """
        Initializes the PointFeatureImportance module.
        :param in_channels: Number of input channels.
        """
        super(PointFeatureImportance, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels, track_running_stats=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.conv(x.transpose(2, 1))
        return x * weight.transpose(2, 1)


class STNkd(nn.Module):
    """
    STNkd module.
    """

    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.k = k

    def forward(self, x):
        x = x.transpose(2, 1)
        x_org = x
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        trans_x = x.view(-1, self.k, self.k)

        x_org = x_org.transpose(2, 1)
        x_org = torch.bmm(x_org, trans_x)
        return x_org