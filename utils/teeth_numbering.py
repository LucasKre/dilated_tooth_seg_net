import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import trimesh
from operator import itemgetter

_teeth_color = {
    8: (129, 175, 129),
    7: (241, 215, 145),
    6: (177, 122, 100),
    5: (111, 184, 210),
    4: (217, 101, 79),
    3: (221, 130, 101),
    2: (144, 239, 144),
    1: (7, 118, 7),
    9: (128, 175, 169),
    10: (242, 215, 185),
    11: (176, 122, 140),
    12: (112, 184, 250),
    13: (218, 101, 119),
    14: (222, 130, 141),
    15: (145, 239, 184),
    16: (8, 118, 47),
    # gum
    0: (204, 204, 204)
}

_teeth_labels = {
    0: 'gum',
    1: 'l_central_incisor',
    2: 'l_lateral_incisor',
    3: 'l_canine',
    4: 'l_1_st_premolar',
    5: 'l_2_nd premolar',
    6: 'l_1_st_molar',
    7: 'l_2_nd_molar',
    8: 'l_3_nd_molar',
    9: 'r_central_incisor',
    10: 'r_lateral_incisor',
    11: 'r_canine',
    12: 'r_1_st_premolar',
    13: 'r_2_nd premolar',
    14: 'r_1_st_molar',
    15: 'r_2_nd_molar',
    17: 'r_3_nd_molar'
}

_teeth_codes_lower = {
    11: (8, 'central_incisor'),
    12: (7, 'lateral_incisor'),
    13: (6, 'canine'),
    14: (5, '1_st_premolar'),
    15: (4, '2_nd premolar'),
    16: (3, '1_st_molar'),
    17: (2, '2_nd_molar'),
    18: (1, '3_nd_molar'),
    21: (9, 'central_incisor'),
    22: (10, 'lateral_incisor'),
    23: (11, 'canine'),
    24: (12, '1_st_premolar'),
    25: (13, '2_nd premolar'),
    26: (14, '1_st_molar'),
    27: (15, '2_nd_molar'),
    28: (16, '3_nd_molar'),
    0: (0, 'gum')
}

_teeth_codes_upper = {
    31: (8, 'central_incisor'),
    32: (7, 'lateral_incisor'),
    33: (6, 'canine'),
    34: (5, '1_st_premolar'),
    35: (4, '2_nd premolar'),
    36: (3, '1_st_molar'),
    37: (2, '2_nd_molar'),
    38: (1, '3_nd_molar'),
    41: (9, 'central_incisor'),
    42: (10, 'lateral_incisor'),
    43: (11, 'canine'),
    44: (12, '1_st_premolar'),
    45: (13, '2_nd premolar'),
    46: (14, '1_st_molar'),
    47: (15, '2_nd_molar'),
    48: (16, '3_nd_molar'),
    0: (0, 'gum')
}

_gum = (0, 'gum')



def color_mesh(mesh: trimesh, labels: np.ndarray) -> trimesh:
    mesh = mesh.copy()
    colors = label_to_colors(labels)
    mesh.visual.face_colors = colors
    return mesh


def fdi_to_label(fdi_codes: np.ndarray) -> np.ndarray:
    teeth_codes = {**_teeth_codes_upper, **_teeth_codes_lower}
    labels = itemgetter(*list(fdi_codes))(teeth_codes)
    return np.array([l[0] for l in labels])


def label_to_colors(labels: np.ndarray) -> np.ndarray:
    sorted_index = sorted(list(_teeth_color.items()), key=lambda tup: tup[0])
    # class_labels = np.array([i[0] for i in sorted_index])
    class_colors = np.array([i[1] for i in sorted_index])

    return class_colors[labels]


def colors_to_label(colors: np.ndarray) -> np.ndarray:
    # ignore alpha channel
    colors = colors[:, :3]
    colors = np.repeat(colors.reshape(-1, 1, 3), 17, axis=1)
    sorted_index = sorted(list(_teeth_color.items()), key=lambda tup: tup[0])
    class_labels = np.array([i[0] for i in sorted_index])
    class_colors = np.array([i[1] for i in sorted_index])
    class_colors = np.repeat(class_colors.reshape(1, -1, 3), colors.shape[0], axis=0)
    mask = (class_colors == colors)
    return class_labels[np.argmax(mask.all(axis=2), axis=1)]
