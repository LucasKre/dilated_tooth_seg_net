import argparse
import glob
import json
import os
import random
import numpy as np

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from dataset.mesh_dataset import MeshTorchDataset3DTeethSeg
from dataset.preprocessing import *
from models.dilated_tooth_seg_net import LitDilatedToothSegNet
from models.dgcnnet import LitDGCNN
from models.meshsegnet import LitMeshSegNet
from models.models import ModelEnum
from models.point_net import LitPointNet
from models.point_net2 import LitPointNet2
from models.tsgcnet import LitTSGCNet

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(SEED)

torch.set_float32_matmul_precision('medium')

random.seed(SEED)

seed_everything(SEED, workers=True)


def get_model(model: ModelEnum, config: dict):
    nr_classes = 17
    pc_size = 16000

    if model == ModelEnum.pointnet:
        kwargs = {"pc_size": pc_size, "classes": nr_classes}
        return LitPointNet(pc_size, nr_classes), kwargs
    elif model == ModelEnum.dgcnn:
        kwargs = {"pc_size": pc_size, "classes": nr_classes}
        return LitDGCNN(pc_size, nr_classes), kwargs
    elif model == ModelEnum.tsgcnet:
        kwargs = {"pc_size": pc_size, "k": 32, "in_channels": 12, "output_channels": nr_classes}
        return LitTSGCNet(pc_size, k=32, in_channels=12, output_channels=nr_classes), kwargs
    elif model == ModelEnum.pointnet2:
        kwargs = {"pc_size": pc_size, "classes": nr_classes}
        return LitPointNet2(pc_size, classes=nr_classes), kwargs
    elif model == ModelEnum.custom_net_2:
        kwargs = {"pc_size": pc_size, "classes": nr_classes, "config": config}
        return LitDilatedToothSegNet(pc_size, nr_classes, config), kwargs
    elif model == ModelEnum.meshsegnet:
        kwargs = {"pc_size": pc_size, "output_channels": nr_classes}
        return LitMeshSegNet(pc_size, output_channels=nr_classes), kwargs
    else:
        raise Exception(f'Model {model} not supported')


def get_dataset(model: ModelEnum, verbose: bool = False, in_memory: bool = True,
                force_process: bool = False, train_test_split=1) -> tuple[Dataset, Dataset]:
    post_transform = None
    if model == ModelEnum.pointnet:
        post_transform = PointNetTransform()
    elif model == ModelEnum.pointnet2:
        post_transform = PointNet2Transform()
    elif model == ModelEnum.pointnet2:
        post_transform = PointNet2Transform()
    elif model == ModelEnum.dgcnn:
        post_transform = DGCNNetTransform()
    elif model == ModelEnum.meshsegnet:
        post_transform = MeshSegNetTransform()
    elif model == ModelEnum.tsgcnet:
        post_transform = TSGCNetTransform()
    elif model == ModelEnum.custom_net_2:
        post_transform = None

    train = MeshTorchDataset3DTeethSeg("data/3dteethseg", processed_folder=f'threedteethseg_preprocessed',
                                       verbose=verbose,
                                       pre_transform=PreTransform(classes=17),
                                       post_transform=post_transform, in_memory=in_memory,
                                       force_process=force_process, is_train=True, train_test_split=train_test_split)
    test = MeshTorchDataset3DTeethSeg("data/3dteethseg", processed_folder=f'threedteethseg_preprocessed',
                                      verbose=verbose,
                                      pre_transform=PreTransform(classes=17),
                                      post_transform=post_transform, in_memory=False,
                                      force_process=False, is_train=False, train_test_split=train_test_split)

    return train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run 3D deep learning experiments')
    parser.add_argument('model', type=lambda m: ModelEnum[m], choices=list(ModelEnum), default=ModelEnum.dgcnn)
    parser.add_argument('config_file', type=str)
    parser.add_argument('--epochs', type=int,
                        help='How many epochs to train', default=10)
    parser.add_argument('--tb_save_dir', type=str,
                        help='Tensorboard save directory', default='tensorboard_logs')
    parser.add_argument('--experiment_name', type=str,
                        help='Experiment Name')
    parser.add_argument('--experiment_version', type=str,
                        help='Experiment Version')
    parser.add_argument('--verbose', action='store_true',
                        help='If log to console')

    parser.add_argument('--validate', action='store_true',
                        help='If should run validation')

    parser.add_argument('--dataset_in_memory', action='store_true',
                        help='If save dataset in memory')
    parser.add_argument('--train_batch_size', type=int,
                        help='Train batch size', default=2)
    parser.add_argument('--devices', nargs='+', help='Devices to use', required=True)
    parser.add_argument('--n_bit_precision', type=int,
                        help='N-Bit precision', default=16)
    parser.add_argument('--force_dataset_process', action='store_true',
                        help='Whether to force preprocessing of dataset')

    parser.add_argument('--skip_training', action='store_true',
                        help='Whether to skip training')

    parser.add_argument('--train_test_split', type=int,
                        help='Train test split option. Either 1 or 2', default=1)

    args = parser.parse_args()

    print(f'Run Experiment using args: {args}')

    # args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.devices)

    with open(os.path.join('experiment_config', args.config_file)) as data:
        config = json.loads(data.read())
        data.close()
        print(f'Using config: {config}')

    train_dataset, test_dataset = get_dataset(args.model, args.verbose, args.dataset_in_memory,
                                              args.force_dataset_process,
                                              args.train_test_split)

    model, kwargs = get_model(args.model, config)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                                   shuffle=True, drop_last=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if args.experiment_name is None:
        experiment_name = f'{args.model}_threedteethseg'
    else:
        experiment_name = args.experiment_name

    experiment_version = args.experiment_version

    logger = TensorBoardLogger(args.tb_save_dir, name=experiment_name, version=experiment_version)

    log_dir = logger.log_dir

    checkpoint_callback = ModelCheckpoint(dirpath=log_dir, save_top_k=1, monitor="val_seg_acc", mode='max')

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='cuda', devices=[int(d) for d in args.devices],
                         enable_progress_bar=args.verbose, logger=logger, precision=args.n_bit_precision,
                         callbacks=[checkpoint_callback], deterministic=False)

    fitted = False
    if not args.skip_training and not os.path.isdir(log_dir):

        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        fitted = True

        trainer.should_stop = True

        best_model_path = os.path.join(*os.path.normpath(checkpoint_callback.best_model_path).split(os.sep)[-4:])

        meta_data = {
            "args": vars(args),
            "seed": SEED,
            "config": config,
            "best_model_path": best_model_path
        }

        with open(os.path.join(logger.log_dir, 'meta_data.json'), 'w') as f:
            for keys in meta_data:
                meta_data[keys] = str(meta_data[keys])
            json.dump(meta_data, f)

    if args.validate and fitted:
        trainer.test(dataloaders=val_dataloader, verbose=args.verbose)
    elif args.validate:
        ckpts = glob.glob(f'{log_dir}/*.ckpt')
        print(f'Loading checkpoint {ckpts[0]} for validation')
        trainer.test(model=model, dataloaders=val_dataloader, ckpt_path=ckpts[0], verbose=args.verbose)
