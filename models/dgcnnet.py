import os
import sys
import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics
import trimesh
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import lightning.pytorch as pl

from utils.teeth_numbering import label_to_colors


def knn(x, k):
    with torch.no_grad():
        x_t = x.transpose(2, 1)
        pairwise_distance = torch.cdist(x_t, x_t, p=2)
        idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx.detach()


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNN(nn.Module):
    def __init__(self, classes, dropout=0.2, k=32):
        super(DGCNN, self).__init__()
        self.k = k
        emb_dims = 1024
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(
            nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(
            nn.Conv1d(1216, 512, kernel_size=1, bias=False),
            self.bn7,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            self.bn8,
            nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Conv1d(256, classes, kernel_size=1, bias=False)

    def forward(self, x):
        bs = x.size(0)
        npoint = x.size(2)

        # (bs, 9, npoint) -> (bs, 9*2, npoint, k)
        x = get_graph_feature(x, k=self.k, dim9=True)
        # (bs, 9*2, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv1(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv2(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # (bs, 64, npoint) -> (bs, 64*2, npoint, k)
        x = get_graph_feature(x1, k=self.k)
        # (bs, 64*2, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv3(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv4(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # (bs, 64, npoint) -> (bs, 64*2, npoint, k)
        x = get_graph_feature(x2, k=self.k)
        # (bs, 64*2, npoint, k) -> (bs, 64, npoint, k)
        x = self.conv5(x)
        # (bs, 64, npoint, k) -> (bs, 64, npoint)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)  # (bs, 64*3, npoint)

        # (bs, 64*3, npoint) -> (bs, emb_dims, npoint)
        x = self.conv6(x)
        # (bs, emb_dims, npoint) -> (bs, emb_dims, 1)
        x = x.max(dim=-1, keepdim=True)[0]

        x = x.repeat(1, 1, npoint)  # (bs, 1024, npoint)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (bs, 1024+64*3, npoint)

        # (bs, 1024+64*3, npoint) -> (bs, 512, npoint)
        x = self.conv7(x)
        # (bs, 512, npoint) -> (bs, 256, npoint)
        x = self.conv8(x)
        x = self.dp1(x)
        # (bs, 256, npoint) -> (bs, 13, npoint)
        x = self.conv9(x)
        x = x.transpose(2, 1).contiguous()
        return x


class LitDGCNN(pl.LightningModule):
    def __init__(self, pc_size, classes, dropout=0.2, k=32):
        super().__init__()
        self.net = DGCNN(classes, dropout=dropout, k=k)
        self.loss_f = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=classes, num_labels=pc_size)
        self.miou = torchmetrics.JaccardIndex(task="multiclass", num_classes=classes, num_labels=pc_size)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=classes,
                                                             num_labels=pc_size)
        self.matthews_c_c = torchmetrics.MatthewsCorrCoef(task="multiclass", num_classes=classes,
                                                          num_labels=pc_size)
        self.dice_score = torchmetrics.Dice(num_classes=classes, average='macro')

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
