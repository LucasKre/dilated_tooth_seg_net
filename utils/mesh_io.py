import os

import trimesh
from os import listdir
from os.path import isfile, join
from utils.teeth_numbering import colors_to_label


def read_mesh(path: str, return_labels: bool = True):
    mesh = trimesh.load(path)
    if return_labels:
        colors = mesh.visual.face_colors
        labels = colors_to_label(colors)
        return mesh, labels
    else:
        return mesh


def iterate_meshes(path: str, return_labels: bool = True, file_format: str = "ply"):
    mesh_files = filter_files(path, file_format)
    for f in mesh_files:
        yield read_mesh(join(path, f), return_labels), f.replace(f'.{file_format}', '')


def filter_files(path: str, file_format: str):
    filtered_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(file_format):
                filtered_files.append(file)
    return filtered_files


def save_mesh(mesh: trimesh, path: str, file_name: str, file_format: str = "ply"):
    mesh.export(f'{path}/{file_name}.{file_format}')
