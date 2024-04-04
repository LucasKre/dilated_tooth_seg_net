import torch
import trimesh


class MoveToOriginTransform(object):
    def __init__(self):
        pass

    def __call__(self, mesh):
        mesh.vertices -= mesh.vertices.mean(axis=0)
        return mesh


class PreTransform(object):
    def __init__(self, classes=17):
        self.classes = classes

    def __call__(self, data):
        mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels = data
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

        pos = x[:, 9:12]

        return pos, x, labels
