import trimesh
import numpy as np
import math
import pandas as pd
import networkx as nx


def pad_array(array, desired_size):
    current_size = array.shape[0]
    pad_width = ((0, max(0, desired_size - current_size)),) + ((0, 0),) * (array.ndim - 1)
    padded_array = np.pad(array, pad_width, mode='constant')
    return padded_array


def face_angels_from_mesh(mesh: trimesh, aggr_func=None):
    means = mesh.vertices[mesh.faces].mean(axis=1)
    normals = mesh.face_normals
    As = means[mesh.face_adjacency]
    vs = normals[mesh.face_adjacency]
    subs = As[:, 0] - As[:, 1]
    sub1s = vs[:, 0] - vs[:, 1]
    scalars = np.asarray(np.sum(subs * sub1s, axis=1))
    mask = scalars > 0
    mask = np.where(mask)
    angels = np.asarray(mesh.face_adjacency_angles.copy())
    angels[mask] = angels[mask] * -1
    angels = angels * 180 / math.pi
    edges = pd.DataFrame(
        {
            "source": mesh.face_adjacency[:, 0],
            "target": mesh.face_adjacency[:, 1],
            "angel": angels,
        }
    )
    graph = nx.from_pandas_edgelist(
        edges,
        edge_attr="angel",
        create_using=nx.Graph(),
    )
    graph = nx.to_dict_of_dicts(graph)
    angels = np.zeros((mesh.faces.shape[0], 3))
    for k in sorted(graph.keys()):
        n = graph[k]
        angels[k] = pad_array(np.array([d["angel"] for d in n.values()]), 3)

    if aggr_func is not None:
        return aggr_func(angels, axis=1)
    return angels
