import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torchmetrics
import trimesh
from torch import nn
from torch.autograd import Variable

from dataset.preprocessing import farthest_point_sample, batched_index_select
from utils.teeth_numbering import label_to_colors


def knn(x, k):
    with torch.no_grad():
        x_t = x.transpose(2, 1)
        pairwise_distance = torch.cdist(x_t, x_t, p=2)
        idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx.detach()


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
    else:
        raise ValueError(
            f'Edge Function {edge_function} is not allowed. Only "global", "local" or "local_global" are valid')

    return feature, idx_org  # (batch_size, 2*num_dims, num_points, k)


class HiddenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0):
        super(HiddenBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_function):
        super(GraphConvBlock, self).__init__()
        self.edge_function = edge_function
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, idx=None, pos=None, k=32):
        x, idx = get_graph_feature(x, edge_function=self.edge_function, k=k, idx=idx, pos=pos)
        x = self.conv(x)
        out = x.max(dim=-1, keepdim=False)[0]
        return out, idx


class STNkd(nn.Module):
    def __init__(self, k=64, norm_track_running_stats: bool = False):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

        self.bn1 = nn.BatchNorm1d(64, track_running_stats=norm_track_running_stats)
        self.bn2 = nn.BatchNorm1d(128, track_running_stats=norm_track_running_stats)
        self.bn3 = nn.BatchNorm1d(1024, track_running_stats=norm_track_running_stats)

        self.k = k

    def forward(self, x):
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

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        trans_x = x.view(-1, self.k, self.k)

        x_org = x_org.transpose(2, 1)
        x_org = torch.bmm(x_org, trans_x)
        x_org = x_org.transpose(2, 1)
        return x_org


class DilatedToothSegNet(nn.Module):
    def __init__(self, classes, config):
        super(DilatedToothSegNet, self).__init__()
        self.ks = config['neighbors']
        self.use_stnkd = config.get('use_stnkd', False)
        self.use_normals = config.get('use_normals', True)
        self.use_angels = config.get('use_angels', True)

        if self.use_stnkd:
            self.stnkd_c = STNkd(k=12)
            self.stnkd_n = STNkd(k=12)

        self.feature_dimensions_1 = 32
        self.feature_dimensions_2 = 48

        '''coordinate feature'''
        self.conv_c_global_1 = GraphConvBlock(12 * 2, self.feature_dimensions_1, self.feature_dimensions_2,
                                              "local_global")
        self.conv_c_global_2 = GraphConvBlock(self.feature_dimensions_2 * 2, self.feature_dimensions_2,
                                              self.feature_dimensions_2,
                                              "local_global")
        self.conv_c_global_3 = GraphConvBlock(self.feature_dimensions_2 * 2, self.feature_dimensions_2,
                                              self.feature_dimensions_2,
                                              "local_global")

        self.conv_local_c = nn.Sequential(
            nn.Conv1d(self.feature_dimensions_2 * 3, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

        if self.use_normals:
            '''normal feature'''
            self.conv_n_global_1 = GraphConvBlock(12, self.feature_dimensions_1, self.feature_dimensions_2,
                                                  "global")

            self.conv_n_global_2 = GraphConvBlock(self.feature_dimensions_2, self.feature_dimensions_2,
                                                  self.feature_dimensions_2,
                                                  "global")

            self.conv_n_global_3 = GraphConvBlock(self.feature_dimensions_2, self.feature_dimensions_2,
                                                  self.feature_dimensions_2,
                                                  "global")
            self.conv_local_n = nn.Sequential(
                nn.Conv1d(self.feature_dimensions_2 * 3, 256, kernel_size=1, bias=False),
                nn.BatchNorm1d(256, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.2)
            )

        if self.use_angels:
            '''angel feature'''
            self.conv_a_global_1 = GraphConvBlock(3, self.feature_dimensions_1, self.feature_dimensions_2,
                                                  "global")

            self.conv_a_global_2 = GraphConvBlock(self.feature_dimensions_2, self.feature_dimensions_2,
                                                  self.feature_dimensions_2,
                                                  "global")

            self.conv_a_global_3 = GraphConvBlock(self.feature_dimensions_2, self.feature_dimensions_2,
                                                  self.feature_dimensions_2,
                                                  "global")

            self.conv_local_a = nn.Sequential(
                nn.Conv1d(self.feature_dimensions_2 * 3, 256, kernel_size=1, bias=False),
                nn.BatchNorm1d(256, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.2)
            )

        self.feature_dim = config['feature_dim']

        local_dim = 256
        if self.use_normals:
            local_dim += 256
        if self.use_angels:
            local_dim += 256

        self.conv_local = nn.Sequential(
            nn.Conv1d(local_dim, self.feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.feature_dim, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.local_global = config['local_global'] != None and len(config['local_global']) > 0
        if self.local_global:
            self.local_global_layers = nn.ModuleList()
            local_feature_dim = self.feature_dim
            for l in config['local_global']:
                self.local_global_layers.append(GraphConvBlock(local_feature_dim * 2, local_feature_dim,
                                                               l["dim"], "local_global"))

                self.feature_dim += l["dim"]

            self.conv_local_global_f = nn.Sequential(
                nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.feature_dim, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.2)
            )

        self.fa = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.feature_dim, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.hidden = nn.Sequential()

        in_dim = self.feature_dim
        for l in config['backbone']:
            self.hidden.append(
                HiddenBlock(in_dim, l['dim'], dropout=0.2))
            in_dim = l['dim']

        self.conv_out = nn.Conv1d(in_dim, classes, kernel_size=1, bias=False)

    def forward(self, x, local_fps_idx):
        npoint = x.size(2)

        c = x[:, :12]
        n = x[:, 12:24]
        a = x[:, 24:]

        if self.use_stnkd:
            c = self.stnkd_c(c)
            n = self.stnkd_n(n)

        c_global_1, idx_c1 = self.conv_c_global_1(c, k=self.ks[0])

        if self.use_normals:
            n_global_1, _ = self.conv_n_global_1(n, idx=idx_c1, k=self.ks[0])

        if self.use_angels:
            a_global_1, _ = self.conv_a_global_1(a, idx=idx_c1, k=self.ks[0])

        c_global_2, idx_c2 = self.conv_c_global_2(c_global_1, k=self.ks[1])

        if self.use_normals:
            n_global_2, _ = self.conv_n_global_2(n_global_1, idx=idx_c2, k=self.ks[1])

        if self.use_angels:
            a_global_2, _ = self.conv_a_global_2(a_global_1, idx=idx_c2, k=self.ks[1])

        c_global_3, idx_c3 = self.conv_c_global_3(c_global_2, k=self.ks[2])

        if self.use_normals:
            n_global_3, _ = self.conv_n_global_3(n_global_2, idx=idx_c3, k=self.ks[2])

        if self.use_angels:
            a_global_3, _ = self.conv_a_global_3(a_global_2, idx=idx_c3, k=self.ks[2])

        c_local = torch.cat((c_global_1, c_global_2, c_global_3), dim=1)
        c_local = self.conv_local_c(c_local)

        if self.use_normals:
            n_local = torch.cat((n_global_1, n_global_2, n_global_3), dim=1)
            n_local = self.conv_local_n(n_local)
        if self.use_angels:
            a_local = torch.cat((a_global_1, a_global_2, a_global_3), dim=1)
            a_local = self.conv_local_a(a_local)

        if self.use_normals and self.use_angels:
            x = torch.cat((c_local, n_local, a_local), dim=1)
        elif self.use_normals:
            x = torch.cat((c_local, n_local), dim=1)
        elif self.use_angels:
            x = torch.cat((c_local, a_local), dim=1)
        else:
            x = c_local

        x = self.conv_local(x)

        local_globals = []

        if self.local_global:
            xl = x
            for i, l in enumerate(self.local_global_layers):
                xl, _ = l(xl, idx=local_fps_idx[:, i], k=local_fps_idx.shape[-1])
                local_globals.append(xl)
            x = torch.cat((x, *local_globals), dim=1)

            x = self.conv_local_global_f(x)

        weight = self.fa(x)
        x = x * weight

        x = self.hidden(x)

        x = self.conv_out(x)

        return x, None


class LitDilatedToothSegNet(pl.LightningModule):
    def __init__(self, pc_size, classes, config):
        super().__init__()
        self.config = config
        self.classes = classes
        self.dist_loss_factor = config.get("dist_loss_factor", 1)
        self.net = DilatedToothSegNet(classes, config)
        self.loss_f = nn.CrossEntropyLoss()
        # self.loss_dist = nn.MSELoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=classes, num_labels=pc_size)
        self.miou = torchmetrics.JaccardIndex(task="multiclass", num_classes=classes, num_labels=pc_size)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=classes,
                                                             num_labels=pc_size)
        self.matthews_c_c = torchmetrics.MatthewsCorrCoef(task="multiclass", num_classes=classes,
                                                          num_labels=pc_size)
        self.dice_score = torchmetrics.Dice(num_classes=classes, average='macro')

    def predict(self, data):
        out = self.predict_prob(data)
        labels_pred = out.max(0)[1]
        return labels_pred

    def predict_prob(self, data):
        x, mesh_triangles, label_dists, labels = data
        local_global_fps_idx = self.get_fps_index(x)
        x = x.transpose(2, 1)
        with torch.no_grad():
            out, attention = self.net(x, local_global_fps_idx)
        out = torch.exp(torch.nn.functional.log_softmax(out, dim=1))[0]
        return out

    def get_fps_index(self, x):
        with torch.no_grad():
            B, N, C = x.shape
            mesh_triangles_mean_c = x[:, :, 9:12]
            local_global = self.config['local_global'] != None and len(self.config['local_global']) > 0
            if local_global:
                idx = []
                cd = torch.cdist(mesh_triangles_mean_c, mesh_triangles_mean_c, p=2)
                for l in self.config['local_global']:
                    r = l["sparse_k"]
                    k = l["k"]
                    idx_t = torch.topk(cd, r, largest=False)[1].reshape(B * N, -1)
                    idx_1 = farthest_point_sample(mesh_triangles_mean_c.reshape(B * N, -1)[idx_t], k)
                    idx_1 = batched_index_select(idx_t, 1, idx_1).reshape(B, N, -1)
                    idx.append(idx_1)
                idx = torch.stack(idx).transpose(0, 1)
            else:
                idx = torch.tensor([])
            return idx

    def validate(self, data):
        x, mesh_triangles, label_dists, labels = data
        local_global_fps_idx = self.get_fps_index(x)
        x = x.transpose(2, 1)
        with torch.no_grad():
            out, dist = self.net(x, local_global_fps_idx)
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
        x, mesh_triangles, label_dists, labels = batch
        local_global_fps_idx = self.get_fps_index(x)
        x = x.transpose(2, 1)
        out, dist = self.net(x, local_global_fps_idx)
        loss = self.loss_f(out, labels)
        sch = self.lr_schedulers()
        if self.trainer.is_last_batch:
            sch.step()
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mesh_triangles, label_dists, labels = batch
        local_global_fps_idx = self.get_fps_index(x)
        x = x.transpose(2, 1)
        out, dist = self.net(x, local_global_fps_idx)
        loss = self.loss_f(out, labels)
        # loss_dist = self.loss_dist(dist, label_dists)
        self.accuracy(out.detach(), labels)
        self.miou(out.detach(), labels)
        self.matthews_c_c(out.detach(), labels)
        self.dice_score(out.detach(), labels)
        self.log('val_seg_acc', self.accuracy, prog_bar=False, logger=True)
        self.log('val_seg_mat_c_c', self.matthews_c_c, prog_bar=False, logger=True)
        self.log('val_miou', self.miou, prog_bar=False, logger=True)
        self.log('val_dice', self.dice_score, prog_bar=False, logger=True)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        # self.log("val_loss_dist", loss_dist, prog_bar=True, logger=True)
        # return loss + (loss_dist * self.dist_loss_factor)
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
        colors_gt = label_to_colors(batch[-1].flatten().detach().cpu().numpy())
        mesh.visual.face_colors = colors_gt
        mesh.export(f'{os.path.join(self.logger.log_dir, "meshes")}/mesh_{batch_idx}_{self.global_rank}_gt.ply')

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
