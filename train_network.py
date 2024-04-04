import argparse
import random
import numpy as np

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data.dataset import Dataset

from dataset.mesh_dataset import Teeth3DSDataset
from dataset.preprocessing import *
from models.dilated_tooth_seg_network import LitDilatedToothSegmentationNetwork

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(SEED)

torch.set_float32_matmul_precision('medium')

random.seed(SEED)

seed_everything(SEED, workers=True)


def get_model():
    return LitDilatedToothSegmentationNetwork()


def get_dataset(train_test_split=1) -> tuple[Dataset, Dataset]:

    train = Teeth3DSDataset("data/3dteethseg", processed_folder=f'processed',
                                       verbose=True,
                                       pre_transform=PreTransform(classes=17),
                                       post_transform=None, in_memory=False,
                                       force_process=False, is_train=True, train_test_split=train_test_split)
    test = Teeth3DSDataset("data/3dteethseg", processed_folder=f'processed',
                                      verbose=True,
                                      pre_transform=PreTransform(classes=17),
                                      post_transform=None, in_memory=False,
                                      force_process=False, is_train=False, train_test_split=train_test_split)

    return train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Training')
    parser.add_argument('--epochs', type=int,
                        help='How many epochs to train', default=100)
    parser.add_argument('--tb_save_dir', type=str,
                        help='Tensorboard save directory', default='tensorboard_logs')
    parser.add_argument('--experiment_name', type=str,
                        help='Experiment Name')
    parser.add_argument('--experiment_version', type=str,
                        help='Experiment Version')
    parser.add_argument('--train_batch_size', type=int,
                        help='Train batch size', default=2)
    parser.add_argument('--devices', nargs='+', help='Devices to use', required=True, default=[0])
    parser.add_argument('--n_bit_precision', type=int,
                        help='N-Bit precision', default=16)
    parser.add_argument('--train_test_split', type=int,
                        help='Train test split option. Either 1 or 2', default=1)
    parser.add_argument('--ckpt', type=str,
                        required=False,
                        help='Checkpoint path to resume training', default=None)


    args = parser.parse_args()

    print(f'Run Experiment using args: {args}')


    train_dataset, test_dataset = get_dataset(args.train_test_split)

    model = get_model()

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

    checkpoint_callback = ModelCheckpoint(dirpath=log_dir, save_top_k=1, monitor="val_acc", mode='max')

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='cuda', devices=[int(d) for d in args.devices],
                         enable_progress_bar=True, logger=logger, precision=args.n_bit_precision,
                         callbacks=[checkpoint_callback], deterministic=False)
    
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.ckpt)
