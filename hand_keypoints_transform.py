import cv2
import numpy as np
import torch
import pyrealsense2 as rs

from ultralytics import YOLO

# from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full

# **Hand Keypoints Connection Pairs for Visualization**
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]


def keypoints_2d_on_image(self, points, cam_trans, focal_length):
    camera_center = [self.width / 2., self.height / 2.]
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = camera_center[0]
    K[1, 2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:]

    V_2d = (K @ points.T).T
    return V_2d[..., :-1]

def calib(self, pos_2d, depth, intrinsics):

    if 0 < pos_2d[0] < self.width and 0 < pos_2d[1] < self.height:
        point_depth = depth.get_distance(round(pos_2d[0]), round(pos_2d[1]))
        ##カメラの内部パラメータからx,yの世界座標を得る
        keypoint = rs.rs2_deproject_pixel_to_point(intrinsics, [round(pos_2d[0]), round(pos_2d[1])], point_depth)
        keypoint = np.array(keypoint)
        return keypoint
    else:
        return None

def distance_between_fingers_normalization(pos1, pos2):
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    v = pos1 - pos2
    v_square = np.square(v)
    l = np.sqrt(np.sum(v_square))
    return l

def coor_trans_from_rs_to_wrs(keypoint_3d_array: np.ndarray):
    if keypoint_3d_array is not None:
        pos = keypoint_3d_array

        ##座標系をwrsの座標系の向きに
        rotmat1 = np.array(
            [[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0], [np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 1]])
        pos_t = pos.T
        pos_t = rotmat1 @ pos_t
        pos = pos_t.T

        # 座標の反転
        pos[0] = -pos[0]
        pos[1] = -pos[1]

        ##平行移動
        pos[0] += 0.38
        pos[1] -= 0.25
        pos[2] += 0.58

        return pos
    else:
        return None

def coor_trans_from_rs_to_xarm(keypoint_3d_array: np.ndarray):
    if keypoint_3d_array is not None:
        pos = keypoint_3d_array

        ##座標系をwrsの座標系の向きに
        rotmat1 = np.array(
            [[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0], [np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 1]])
        pos_t = pos.T
        pos_t = rotmat1 @ pos_t
        pos = pos_t.T


        ##平行移動
        pos[0] += 0.4
        pos[1] -= 0.3
        pos[2] += 0.0

        return pos
    else:
        return None

def parse_keypoint_3d(all_joints) -> np.ndarray:
    keypoint_3d = np.empty([21, 3])
    for i in range(21):
        keypoint_3d[i] = np.ndarray(all_joints[i])
    return keypoint_3d

def rotmat_to_xarm(verts) -> np.ndarray:
    np_verts = np.array(verts)

    ##WRS上の座標系に対する手の向きの変換
    convert = np.array(
        [[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0], [np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 1]])

    ##手の上の姿勢変換
    # rotmat1 = np.array([[np.cos(np.pi), 0, np.sin(np.pi)], [0, 1, 0], [-np.sin(np.pi), 0, np.cos(np.pi)]])
    rotmat2 = np.array(
        [[1, 0, 0], [0, np.cos(-np.pi / 2), -np.sin(-np.pi / 2)], [0, np.sin(-np.pi / 2), np.cos(-np.pi / 2)]])

    # np_verts = convert @ np_verts @ rotmat1 @ rotmat2

    np_verts = convert @ np_verts @ rotmat2
    #
    # np_verts = np_verts @ rotmat2

    return np_verts
