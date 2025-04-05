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


class HandDetector_multifinger:
    def __init__(self, device, model, model_cfg, renderer, detector, hand_type="Right", detect_hand_num=1, WIDTH=640,
                 HEIGHT=480):
        self.device = device
        self.model = model
        self.model_cfg = model_cfg
        self.renderer = renderer
        self.detector = detector
        self.detected_hand_type = hand_type
        self.detect_hand_num = detect_hand_num
        self.width = WIDTH
        self.height = HEIGHT

    def render_reconstruction(self, input_img, num_dets, reconstructions):
        if num_dets > 0:
            # Render front view
            misc_args = dict(
                mesh_base_color=(0.95098039, 0.274117647, 0.65882353),
                scene_bg_color=(1, 1, 1),
                focal_length=reconstructions['focal'],
            )

            cam_view = self.renderer.render_rgba_multiple(reconstructions['verts'],
                                                          cam_t=reconstructions['cam_t'],
                                                          render_res=reconstructions['img_size'],
                                                          is_right=reconstructions['right'], **misc_args)

            # Overlay image

            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)  # Add alpha channel
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

            # cv2.circle(input_img_overlay,
            #            (int(reconstructions['kpts_2d'][0][0]), int(reconstructions['kpts_2d'][0][1])), 5,
            #            (0, 0, 255), -1)

            return input_img_overlay, f'{num_dets} hands detected'
        else:
            return input_img, f'{num_dets} hands detected'

    def run_wilow_model(self, image, IoU_threshold=0.3):
        img_cv2 = image[:, :, ::-1]  ##RGBをBGRの順に変換
        img_vis = image.copy()  ##入力画像をそのままコピー

        detections = self.detector(img_cv2, verbose=False, iou=IoU_threshold)[0]

        bboxes = []
        is_right = []
        detected_hand_count = 0
        for det in detections:
            if det.boxes.cls.data.cpu().detach().item() == 1 and detected_hand_count < 1:
                detected_hand_count += 1
                Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                Conf = det.boxes.conf.data.cpu().detach()[0].numpy().reshape(-1).astype(np.float16)  ##IoUのしきい値
                Side = det.boxes.cls.data.cpu().detach()  ##右手と左手　(左手:1 右手:0)

                is_right.append(det.boxes.cls.cpu().detach().squeeze().item())  ##画面に入っている手の数を右手と左手でわけて追加
                bboxes.append(Bbox[:4].tolist())  ##検出した手を囲む枠の左上のピクセルと右下のピクセル

                color = (255 * 0.208, 255 * 0.647, 255 * 0.603) if Side == 0. else (
                    255 * 1, 255 * 0.78039, 255 * 0.2353)  ##右手と左手で色の使い分け
                label = f'L - {Conf[0]:.3f}' if Side == 0 else f'R - {Conf[0]:.3f}'

                cv2.rectangle(img_vis, (int(Bbox[0]), int(Bbox[1])), (int(Bbox[2]), int(Bbox[3])), color, 3)  ##枠の表示
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img_vis, (int(Bbox[0]), int(Bbox[1]) - 20), (int(Bbox[0]) + w, int(Bbox[1])), color,
                              -1)  ##テキストのフレーム
                cv2.putText(img_vis, label, (int(Bbox[0]), int(Bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                            2)
            else:
                continue

        if len(bboxes) != 0:
            boxes = np.stack(bboxes)
            right = np.stack(is_right)  ##認識された順に右手と左手のラベルが格納されている

            dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            all_verts = []
            all_cam_t = []
            all_right = []
            all_joints_points = []  ##関節位置の格納

            for batch in dataloader:
                batch = recursive_to(batch, self.device)

                with torch.no_grad():
                    out = self.model(batch)

                multiplier = (2 * batch['right'] - 1)  ##左手:-1 右手:+1
                pred_cam = out['pred_cam']
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()  ##画面の大きさ
                scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                                   scaled_focal_length).detach().cpu().numpy()

                batch_size = batch['img'].shape[0]  ##画面に写っている手の数
                for n in range(batch_size):
                    verts = out['pred_vertices'][n].detach().cpu().numpy()  ##多分姿勢
                    joints_points = out['pred_keypoints_3d'][n].detach().cpu().numpy()  ##関節の位置21個
                    rotmat = out['pred_mano_params']['global_orient'][n].detach().cpu().numpy()

                    is_right = batch['right'][n].cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                    joints_points[:, 0] = (2 * is_right - 1) * joints_points[:, 0]
                    cam_t = pred_cam_t_full[n]  ##　？

                    kpts_2d = self.keypoints_2d_on_image(joints_points, cam_t, scaled_focal_length)
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)
                    all_joints_points.append(joints_points)

            reconstructions = {'verts': all_verts, 'cam_t': all_cam_t, 'right': all_right, 'img_size': img_size[n],
                               'focal': scaled_focal_length, 'joints_points': all_joints_points,
                               'box_center': box_center, 'rotmat': rotmat, 'kpts_2d': kpts_2d}

            return img_vis.astype(np.float32) / 255.0, detected_hand_count, reconstructions
        else:
            return img_vis.astype(np.float32) / 255.0, detected_hand_count, None

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

    @staticmethod
    def distance_between_fingers_normalization(pos1, pos2):
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        v = pos1 - pos2
        v_square = np.square(v)
        l = np.sqrt(np.sum(v_square))
        return l

    @staticmethod
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

    @staticmethod
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
            pos[2] -= 0.5

            return pos
        else:
            return None

    @staticmethod
    def parse_keypoint_3d(all_joints) -> np.ndarray:
        keypoint_3d = np.empty([21, 3])
        for i in range(21):
            keypoint_3d[i] = np.ndarray(all_joints[i])
        return keypoint_3d

    @staticmethod
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

    @staticmethod
    def compute_angle(v1, v2):
        """
        Compute the angle between two vectors using the cosine formula.
        :param v1: First vector (numpy array)
        :param v2: Second vector (numpy array)
        :return: Angle in degrees
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:  # Avoid division by zero
            return 0.0
        cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        return np.arccos(cos_theta)

    @staticmethod
    def map_angle(x, src_min, src_max, dst_min, dst_max, reverse=True):
        if x > src_max:
            x = src_max
        if x < src_min:
            x = src_min
        if reverse:
            return (x - src_min) / (src_max - src_min) * (dst_min - dst_max) + dst_max
        else:
            return (x - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min

    def compute_selected_angles(self, keypoints):
        """
        Compute angles for selected joints in the hand skeleton.
        :param keypoints: 21 hand keypoints from WiLor Model
        :return: Dictionary of computed angles
        """
        joint_pairs = {
            "thumb0": (0, 1, 2),
            "thumb1": (1, 2, 3),
            "thumb2": (2, 3, 4),
            "index0": (10, 5, 6),
            "index1": (0, 5, 6),
            "index2": (5, 6, 7),
            "middle0": (0, 9, 10),
            "middle1": (9, 10, 11),
            "ring0": (0, 13, 14),
            "ring1": (13, 14, 15),
            "pinky0": (0, 17, 18),
            "pinky1": (17, 18, 19)
        }
        angles = {}
        for joint_name, (p1, p2, p3) in joint_pairs.items():
            v1 = np.array(keypoints[p1]) - np.array(keypoints[p2])  # Vector 1
            v2 = np.array(keypoints[p3]) - np.array(keypoints[p2])  # Vector 2
            angles[joint_name] = self.compute_angle(v1, v2)
            if joint_name == "thumb0":
                angles[joint_name] = self.map_angle(angles[joint_name], 2.4, 2.9, 0, 1.57)
            if joint_name == "thumb1":
                angles[joint_name] = self.map_angle(angles[joint_name], 2.5, 3.0, 0, 1.0)
            if joint_name == "thumb2":
                angles[joint_name] = self.map_angle(angles[joint_name], 2.3, 2.9, 0, 1.57)
            if joint_name == "index0":
                angles[joint_name] = self.map_angle(angles[joint_name], 0.5, 1.0, -0.04, 0.297, reverse=False)
            if joint_name == "index1":
                angles[joint_name] = self.map_angle(angles[joint_name], 2.0, 2.9, 0, 1.5)
            if joint_name == "index2":
                angles[joint_name] = self.map_angle(angles[joint_name], 1.6, 3, 0, 1.92)
            if joint_name == "middle0":
                angles[joint_name] = self.map_angle(angles[joint_name], 2.0, 2.9, 0, 1.5)
            if joint_name == "middle1":
                angles[joint_name] = self.map_angle(angles[joint_name], 1.6, 2.9, 0, 1.92)
            if joint_name == "ring0":
                angles[joint_name] = self.map_angle(angles[joint_name], 2.0, 2.9, 0, 1.92)
            if joint_name == "ring1":
                angles[joint_name] = self.map_angle(angles[joint_name], 1.6, 2.8, 0, 1.92)
            if joint_name == "pinky0":
                angles[joint_name] = self.map_angle(angles[joint_name], 2.0, 2.9, 0, 1.92)
            if joint_name == "pinky1":
                angles[joint_name] = self.map_angle(angles[joint_name], 1.6, 2.9, 0, 1.92)
        return angles