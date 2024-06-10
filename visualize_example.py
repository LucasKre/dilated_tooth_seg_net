import argparse
import random
import numpy as np
from pathlib import Path 
import trimesh
from utils.teeth_numbering import color_mesh

from lightning.pytorch import seed_everything
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


def get_dataset(train_test_split=1) -> Dataset:

    test = Teeth3DSDataset("data/3dteethseg", processed_folder=f'processed',
                                      verbose=True,
                                      pre_transform=PreTransform(classes=17),
                                      post_transform=None, in_memory=False,
                                      force_process=False, is_train=False, train_test_split=train_test_split)

    return test


def infer(ckpt_path, train_test_split=1, data_idx=0, save_mesh=False, out_dir='plots', return_scene=False, use_gpu=True):
    print(f"Running inference on data index {data_idx} using checkpoint {ckpt_path} with train_test_split {train_test_split}. Use GPU: {use_gpu}")
    test_dataset = get_dataset(train_test_split)

    model = LitDilatedToothSegmentationNetwork.load_from_checkpoint(ckpt_path)

    if use_gpu:
        model = model.cuda()

    data = test_dataset[data_idx]
    triangles = data[1][:, :9].reshape(-1, 3, 3)
    mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(triangles.cpu().detach().numpy()))
    ground_truth = data[2]
    pre_labels = model.predict_labels(data).cpu().numpy()
    mesh_pred = color_mesh(mesh, pre_labels)
    mesh_gt = color_mesh(mesh, ground_truth)
    if save_mesh:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        mesh_pred.export(f'{out_dir}/pred_{data_idx}.ply')
        mesh_gt.export(f'{out_dir}/gt_{data_idx}.ply')
    if return_scene:
        mesh_gt.vertices += np.array([5, 0, 0])
        scene = trimesh.Scene([mesh_pred, mesh_gt])
        return scene




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run visualization of an example from the dataset')
    parser.add_argument('--out_dir', type=str,
                        help='Output directory where the mesh will be saved', default='predictions')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--n_bit_precision', type=int,
                        help='N-Bit precision', default=16)
    parser.add_argument('--train_test_split', type=int,
                        help='Train test split option. Either 1 or 2', default=1)
    parser.add_argument('--data_idx', type=int, default=0)
    parser.add_argument('--save_mesh', type=bool, default=False)
    parser.add_argument('--ckpt', type=str,
                        required=True,
                        help='Checkpoint path')

    args = parser.parse_args()
    
    infer(args.ckpt, args.train_test_split, args.data_idx, args.save_mesh, args.out_dir, use_gpu=args.use_gpu)
    

    
