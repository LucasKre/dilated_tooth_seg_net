import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics
import trimesh
from torch import optim
from torch.autograd import Variable

from utils.teeth_numbering import label_to_colors


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


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
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNet(nn.Module):
    def __init__(self, num_class):
        super(PointNet, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        return x


class LitPointNet(pl.LightningModule):
    def __init__(self, pc_size, classes=2):
        super().__init__()
        self.net = PointNet(classes)
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


if __name__ == '__main__':
    x = torch.rand(2, 3, 10000, requires_grad=True).cuda()
    label = torch.rand(2, 10000, 12).float().cuda()
    loss_f = nn.CrossEntropyLoss()
    net = PointNet(12).cuda()
    net.train()
    out = net(x)
    loss = loss_f(out, label)
    loss.backward()
    for name, param in net.named_parameters():
        if param.grad is None:
            print(name)
