import os
from pathlib import Path

import numpy as np
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import trimesh
from torch import optim
from torch.autograd import Variable

from utils.teeth_numbering import label_to_colors


def knn(x, k):
    with torch.no_grad():
        x_t = x.transpose(2, 1)
        pairwise_distance = torch.cdist(x_t, x_t, p=2)
        idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx.detach()


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


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def get_graph_feature(coor, nor, k=10):
    batch_size, num_dims, num_points = coor.shape
    coor = coor.view(batch_size, -1, num_points)

    idx = knn(coor, k=k)
    index = idx
    device = coor.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = coor.size()
    _, num_dims2, _ = nor.size()

    coor = coor.transpose(2, 1).contiguous()
    nor = nor.transpose(2, 1).contiguous()

    # coordinate
    coor_feature = coor.view(batch_size * num_points, -1)[idx, :]
    coor_feature = coor_feature.view(batch_size, num_points, k, num_dims)
    coor = coor.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    coor_feature = torch.cat((coor_feature, coor), dim=3).permute(0, 3, 1, 2).contiguous()

    # normal vector
    nor_feature = nor.view(batch_size * num_points, -1)[idx, :]
    nor_feature = nor_feature.view(batch_size, num_points, k, num_dims2)
    nor = nor.view(batch_size, num_points, 1, num_dims2).repeat(1, 1, k, 1)
    nor_feature = torch.cat((nor_feature, nor), dim=3).permute(0, 3, 1, 2).contiguous()
    return coor_feature, nor_feature, index


class GraphAttention(nn.Module):
    def __init__(self, feature_dim, out_dim, K):
        super(GraphAttention, self).__init__()
        self.dropout = 0.6
        self.conv = nn.Sequential(nn.Conv2d(feature_dim * 2, out_dim, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(out_dim),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.K = K

    def forward(self, Graph_index, x, feature):
        B, C, N = x.shape
        x = x.contiguous().view(B, N, C)
        feature = feature.permute(0, 2, 3, 1)
        neighbor_feature = index_points(x, Graph_index)
        centre = x.view(B, N, 1, C).expand(B, N, self.K, C)
        delta_f = torch.cat([centre - neighbor_feature, neighbor_feature], dim=3).permute(0, 3, 2, 1)
        e = self.conv(delta_f)
        e = e.permute(0, 3, 2, 1)
        attention = F.softmax(e, dim=2)  # [B, npoint, nsample,D]
        graph_feature = torch.sum(torch.mul(attention, feature), dim=2).permute(0, 2, 1)
        return graph_feature


class TSGCNet(nn.Module):
    def __init__(self, k=16, in_channels=12, output_channels=8):
        super(TSGCNet, self).__init__()
        self.k = k
        ''' coordinate stream '''
        self.bn1_c = nn.BatchNorm2d(64)
        self.bn2_c = nn.BatchNorm2d(128)
        self.bn3_c = nn.BatchNorm2d(256)
        self.bn4_c = nn.BatchNorm1d(512)
        self.conv1_c = nn.Sequential(nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False),
                                     self.bn1_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv2_c = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                     self.bn2_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv3_c = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                     self.bn3_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv4_c = nn.Sequential(nn.Conv1d(448, 512, kernel_size=1, bias=False),
                                     self.bn4_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.attention_layer1_c = GraphAttention(feature_dim=12, out_dim=64, K=self.k)
        self.attention_layer2_c = GraphAttention(feature_dim=64, out_dim=128, K=self.k)
        self.attention_layer3_c = GraphAttention(feature_dim=128, out_dim=256, K=self.k)
        self.FTM_c1 = STNkd(k=12)
        ''' normal stream '''
        self.bn1_n = nn.BatchNorm2d(64)
        self.bn2_n = nn.BatchNorm2d(128)
        self.bn3_n = nn.BatchNorm2d(256)
        self.bn4_n = nn.BatchNorm1d(512)
        self.conv1_n = nn.Sequential(nn.Conv2d((in_channels) * 2, 64, kernel_size=1, bias=False),
                                     self.bn1_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv2_n = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                     self.bn2_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv3_n = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                     self.bn3_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv4_n = nn.Sequential(nn.Conv1d(448, 512, kernel_size=1, bias=False),
                                     self.bn4_n,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.FTM_n1 = STNkd(k=12)

        '''feature-wise attention'''

        self.fa = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                nn.BatchNorm1d(1024),
                                nn.LeakyReLU(0.2))

        ''' feature fusion '''
        self.pred1 = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred4 = nn.Sequential(nn.Conv1d(128, output_channels, kernel_size=1, bias=False))
        self.dp1 = nn.Dropout(p=0.6)
        self.dp2 = nn.Dropout(p=0.6)
        self.dp3 = nn.Dropout(p=0.6)

    def forward(self, x):
        B, C, N = x.shape
        coor = x[:, :12, :]
        nor = x[:, 12:, :]

        # transform
        trans_c = self.FTM_c1(coor)
        coor = coor.transpose(2, 1)
        coor = torch.bmm(coor, trans_c)
        coor = coor.transpose(2, 1)
        trans_n = self.FTM_n1(nor)
        nor = nor.transpose(2, 1)
        nor = torch.bmm(nor, trans_n)
        nor = nor.transpose(2, 1)

        coor1, nor1, index = get_graph_feature(coor, nor, k=self.k)
        coor1 = self.conv1_c(coor1)
        nor1 = self.conv1_n(nor1)
        coor1 = self.attention_layer1_c(index, coor, coor1)
        nor1 = nor1.max(dim=-1, keepdim=False)[0]

        coor2, nor2, index = get_graph_feature(coor1, nor1, k=self.k)
        coor2 = self.conv2_c(coor2)
        nor2 = self.conv2_n(nor2)
        coor2 = self.attention_layer2_c(index, coor1, coor2)
        nor2 = nor2.max(dim=-1, keepdim=False)[0]

        coor3, nor3, index = get_graph_feature(coor2, nor2, k=self.k)
        coor3 = self.conv3_c(coor3)
        nor3 = self.conv3_n(nor3)
        coor3 = self.attention_layer3_c(index, coor2, coor3)
        nor3 = nor3.max(dim=-1, keepdim=False)[0]

        coor = torch.cat((coor1, coor2, coor3), dim=1)
        coor = self.conv4_c(coor)
        nor = torch.cat((nor1, nor2, nor3), dim=1)
        nor = self.conv4_n(nor)

        avgSum_coor = coor.sum(1) / 512
        avgSum_nor = nor.sum(1) / 512
        avgSum = avgSum_coor + avgSum_nor
        weight_coor = (avgSum_coor / avgSum).reshape(B, 1, N)
        weight_nor = (avgSum_nor / avgSum).reshape(B, 1, N)
        x = torch.cat((coor * weight_coor, nor * weight_nor), dim=1)

        weight = self.fa(x)
        x = weight * x

        x = self.pred1(x)
        self.dp1(x)
        x = self.pred2(x)
        self.dp2(x)
        x = self.pred3(x)
        self.dp3(x)
        score = self.pred4(x)
        score = F.log_softmax(score, dim=1)
        score = score.permute(0, 2, 1)
        return score


class LitTSGCNet(pl.LightningModule):
    def __init__(self, pc_size, k=16, in_channels=12, output_channels=8):
        super().__init__()
        self.net = TSGCNet(k, in_channels, output_channels)
        self.loss_f = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_channels, num_labels=pc_size)
        self.miou = torchmetrics.JaccardIndex(task="multiclass", num_classes=output_channels, num_labels=pc_size)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=output_channels,
                                                             num_labels=pc_size)
        self.matthews_c_c = torchmetrics.MatthewsCorrCoef(task="multiclass", num_classes=output_channels,
                                                          num_labels=pc_size)
        self.dice_score = torchmetrics.Dice(num_classes=output_channels, average='macro')

    def predict(self, data):
        x, mesh_triangles, labels = data
        x = x.transpose(2, 1)
        with torch.no_grad():
            out = self.net(x)
        out = torch.exp(torch.nn.functional.log_softmax(out.transpose(2, 1), dim=1))[0]
        labels_pred = out.max(0)[1]
        return labels_pred

    def validate(self, data):
        x, mesh_triangles, labels = data
        x = x.transpose(2, 1)
        with torch.no_grad():
            out = self.net(x)
        out = out.transpose(2, 1)
        self.accuracy(out, labels)
        self.miou(out, labels)
        self.matthews_c_c(out, labels)
        self.dice_score(out, labels)
        metrics = {
            'final_accuracy': self.accuracy.compute(),
            'final_miou': self.miou.compute(),
            'final_dice_score': self.dice_score.compute(),
            'final_matthews_c_c': self.matthews_c_c.compute()
        }
        self.accuracy.reset()
        self.miou.reset()
        self.dice_score.reset()
        self.matthews_c_c.reset()
        return metrics

    def training_step(self, batch, batch_idx):
        x, mesh_triangles, labels = batch
        x = x.transpose(2, 1)
        out = self.net(x)
        out = out.transpose(2, 1)
        loss = self.loss_f(out, labels)
        sch = self.lr_schedulers()
        # step every N epochs
        if self.trainer.is_last_batch:
            sch.step()
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mesh_triangles, labels = batch
        x = x.transpose(2, 1)
        out = self.net(x)
        out = out.transpose(2, 1)
        loss = self.loss_f(out, labels)
        self.accuracy(out.detach(), labels)
        self.miou(out.detach(), labels)
        self.matthews_c_c(out.detach(), labels)
        self.dice_score(out.detach(), labels)
        self.log('val_seg_acc', self.accuracy, prog_bar=False, logger=True)
        self.log('val_seg_mat_c_c', self.matthews_c_c, prog_bar=False, logger=True)
        self.log('val_miou', self.miou, prog_bar=False, logger=True)
        self.log('val_dice', self.dice_score, prog_bar=False, logger=True)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        metrics = self.validate(batch)
        self.log_dict(metrics)
        labels_pred = self.predict(batch).cpu().numpy()
        mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(batch[1][0].cpu().detach().numpy()))
        colors = label_to_colors(labels_pred)
        mesh.visual.face_colors = colors
        mesh_path = f'{os.path.join(self.logger.log_dir, "meshes")}'
        Path(mesh_path).mkdir(parents=True, exist_ok=True)
        mesh.export(f'{os.path.join(self.logger.log_dir, "meshes")}/mesh_{batch_idx}_{self.global_rank}.ply')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=60, gamma=0.5, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
            }
        }
