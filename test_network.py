import argparse
import random
import numpy as np

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
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


def get_dataset(train_test_split=1) -> Dataset:

    test = Teeth3DSDataset("data/3dteethseg", processed_folder=f'processed',
                                      verbose=True,
                                      pre_transform=PreTransform(classes=17),
                                      post_transform=None, in_memory=False,
                                      force_process=False, is_train=False, train_test_split=train_test_split)

    return test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run testing')
    parser.add_argument('--epochs', type=int,
                        help='How many epochs to train', default=10)
    parser.add_argument('--tb_save_dir', type=str,
                        help='Tensorboard save directory', default='tensorboard_logs')
    parser.add_argument('--experiment_name', type=str,
                        help='Experiment Name')
    parser.add_argument('--experiment_version', type=str,
                        help='Experiment Version')
    parser.add_argument('--train_batch_size', type=int,
                        help='Train batch size', default=2)
    parser.add_argument('--devices', nargs='+', help='Devices to use', required=True)
    parser.add_argument('--n_bit_precision', type=int,
                        help='N-Bit precision', default=16)
    parser.add_argument('--train_test_split', type=int,
                        help='Train test split option. Either 1 or 2', default=1)
    parser.add_argument('--ckpt', type=str,
                        required=True,
                        help='Checkpoint path')

    args = parser.parse_args()

    print(f'Run Experiment using args: {args}')


    test_dataset = get_dataset(args.train_test_split)

    model = get_model()

    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if args.experiment_name is None:
        experiment_name = f'{args.model}_threedteethseg'
    else:
        experiment_name = args.experiment_name

    experiment_version = args.experiment_version

    logger = CSVLogger(args.tb_save_dir, name=experiment_name, version=experiment_version)

    log_dir = logger.log_dir


    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='cuda', devices=[int(d) for d in args.devices],
                         enable_progress_bar=True, logger=logger, precision=args.n_bit_precision,  deterministic=False)
    
    trainer.test(model=model, dataloaders=val_dataloader, ckpt_path=args.ckpt, verbose=True)
