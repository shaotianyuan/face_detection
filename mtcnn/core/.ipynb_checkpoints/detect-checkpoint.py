import cv2
import time
import numpy as np
import torch

from models import PNet


def creat_mtcnn_net(p_model_path=None, r_model_path=None, o_model_path=None):
    pnet, rnet, onet = None, None, None

    if p_model_path:
        pnet = PNet()
        pnet.load_state_dict(torch.load(p_model_path))
        pnet.eval()

    if r_model_path:
        rnet = RNet()
        rnet.load_state_dict(torch.load(r_model_path))
        rnet.eval()

    if o_model_path:
        onet = ONet()
        onet.load_state_dict(torch.load(o_model_path))
        onet.eval()

    return pnet, rnet, onet
