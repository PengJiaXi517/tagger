import os

import numpy as np

def trans_odom2ego(x, y, rot_theta, path):

    rot_mat = np.array(
        [
            [
                np.cos(rot_theta),
                -np.sin(rot_theta),
            ],
            [
                np.sin(rot_theta),
                np.cos(rot_theta),
            ],
        ]
    ).T
    t_vec = -np.array([x, y]) @ rot_mat.T
    path[..., :2] = path[..., :2] @ rot_mat.T + t_vec

    return path

def trans_ego2odom():
    pass