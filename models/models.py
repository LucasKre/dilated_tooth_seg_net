from enum import Enum
from models.point_net import LitPointNet
from models.dgcnnet import LitDGCNN
from models.tsgcnet import LitTSGCNet
from models.dilated_tooth_seg_net import LitDilatedToothSegNet
from models.point_net2 import LitPointNet2
from models.meshsegnet import LitMeshSegNet


class ModelEnum(Enum):
    dgcnn = ('dgcnn', LitDGCNN)
    pointnet = ('pointnet', LitPointNet)
    tsgcnet = ('tsgcnet', LitTSGCNet)
    pointnet2 = ('pointnet2', LitPointNet2)
    custom_net_2 = ('custom_net_2', LitDilatedToothSegNet)
    meshsegnet = ('meshsegnet', LitMeshSegNet)

    def __str__(self):
        return self.value[0]
    
    def __repr__(self):
        return str(self)
