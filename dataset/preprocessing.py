import torch
import trimesh


def normalize(x: torch.tensor):
    centroid = torch.mean(x, dim=0)
    x -= centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(abs(x) ** 2, dim=-1)))
    x /= furthest_distance
    return x


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def compute_dist_to_next_label(vertices, labels):
    dist_m = torch.zeros(len(labels)).float()
    vertices_centroid = vertices.mean(dim=1)
    for label_idx in torch.unique(labels):
        idxs = torch.argwhere(labels == label_idx)
        vertices_of_int = vertices[labels == label_idx]
        vertices_of_int_centroid = vertices_of_int.mean(dim=1)
        dist = torch.cdist(vertices_of_int_centroid, vertices_centroid[labels != label_idx], p=2)
        min_dist = dist.min(dim=1)[0].float()
        dist_m[idxs.flatten()] = min_dist
    return dist_m


class Compose(object):
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class MoveToOriginTransform(object):
    def __init__(self):
        pass

    def __call__(self, mesh):
        mesh.vertices -= mesh.vertices.mean(axis=0)
        return mesh


class DGCNNetTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        xi, mesh_triangles, label_dist, labels = data
        x = torch.zeros(xi.shape[0], 6).float()
        x[:, :3] = xi[:, 9:12]
        x[:, 3:6] = xi[:, 21: 24]

        return x, mesh_triangles, labels


class TSGCNetTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        xi, mesh_triangles, label_dists, labels = data
        return xi, mesh_triangles, labels


class PointNetTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        xi, mesh_triangles, label_dists, labels = data
        x = torch.zeros(xi.shape[0], 3).float()
        x[:, :3] = xi[:, 9:12]
        return x, mesh_triangles, labels


class PointNet2Transform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        xi, mesh_triangles, label_dists, labels = data
        x = torch.zeros(xi.shape[0], 6).float()
        x[:, :3] = xi[:, 9:12]
        x[:, 3:6] = xi[:, 21: 24]

        return x, mesh_triangles, labels


class MeshSegNetTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        xi, mesh_triangles, label_dists, labels = data
        x = torch.zeros(xi.shape[0], 15).float()
        x[:, :12] = xi[:, :12]
        x[:, 12:15] = xi[:, 21: 24]

        return x, mesh_triangles, labels


class PreTransform(object):
    def __init__(self, classes=17):
        self.classes = classes

    def __call__(self, data):
        mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels, angels = data
        mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(mesh_triangles.cpu().detach().numpy()))

        points = torch.from_numpy(mesh.vertices)
        v_normals = torch.from_numpy(mesh.vertex_normals)

        s, _ = mesh_faces.size()
        x = torch.zeros(s, 24).float()
        x[:, :3] = mesh_triangles[:, 0]
        x[:, 3:6] = mesh_triangles[:, 1]
        x[:, 6:9] = mesh_triangles[:, 2]
        x[:, 9:12] = mesh_triangles.mean(dim=1)
        x[:, 12:15] = mesh_vertices_normals[:, 0]
        x[:, 15:18] = mesh_vertices_normals[:, 1]
        x[:, 18:21] = mesh_vertices_normals[:, 2]
        x[:, 21:] = mesh_face_normals

        maxs = points.max(dim=0)[0]
        mins = points.min(dim=0)[0]
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = v_normals.mean(axis=0)
        nstds = v_normals.std(axis=0)
        nmeans_f = mesh_face_normals.mean(axis=0)
        nstds_f = mesh_face_normals.std(axis=0)
        for i in range(3):
            # normalize coordinate
            x[:, i] = (x[:, i] - means[i]) / stds[i]  # point 1
            x[:, i + 3] = (x[:, i + 3] - means[i]) / stds[i]  # point 2
            x[:, i + 6] = (x[:, i + 6] - means[i]) / stds[i]  # point 3
            x[:, i + 9] = (x[:, i + 9] - mins[i]) / (maxs[i] - mins[i])  # centre
            # normalize normal vector
            x[:, i + 12] = (x[:, i + 12] - nmeans[i]) / nstds[i]  # normal1
            x[:, i + 15] = (x[:, i + 15] - nmeans[i]) / nstds[i]  # normal2
            x[:, i + 18] = (x[:, i + 18] - nmeans[i]) / nstds[i]  # normal3
            x[:, i + 21] = (x[:, i + 21] - nmeans_f[i]) / nstds_f[i]  # face normal

        label_dists = normalize(compute_dist_to_next_label(mesh_triangles, labels))

        return x, mesh_triangles, label_dists, labels
