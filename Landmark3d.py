from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
# from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, cv_draw_landmark, get_suffix

import torch
from torch import nn


class Landmark3dModel(nn.Module):
    def __init__(self, device='cpu', **config):
        super(Landmark3dModel, self).__init__()
        
        print("Loading Landmark3dModel to...", device)
        self.face_boxes = FaceBoxes()
        mode = True if device == 'cuda' else False
        self.tddfa = TDDFA(gpu_mode=False, **config)
        
    def forward(self, x, pose=False, no_grad=True):
        boxes = self.face_boxes(x)
#         print(boxes)

        # regress 3DMM params
        #
        if no_grad:
            with torch.no_grad():
                param_lst, roi_box_lst = self.tddfa(x, boxes)
        else:
            param_lst, roi_box_lst = self.tddfa(x, boxes)

        if pose:
            print("!!! Not implemented - need to return pose params !!!")
            
        # reconstruct vertices and visualizing sparse landmarks
        #
        dense_flag = False
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        landmarks_3d = ver_lst[0].T
#         print(gt_landmarks_3d.shape)
        return landmarks_3d
        
        