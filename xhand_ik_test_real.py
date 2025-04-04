from hand_detector_multi_finger import HandDetector_multifinger
from xhand_class_ikpy import xhand_K
from utils.data_class import animation, WiLor_Data, Data
from utils.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

import pickle
from pathlib import Path
import pyrealsense2 as rs
import multiprocessing
import torch
import time
import cv2
import numpy as np
from queue import Empty
from loguru import logger
from wrs import wd, rm, ur3d, rrtc, mgm, mcm
from wrs.robot_sim.end_effectors.multifinger.xhand import xhand_right as xhr
from wrs.robot_con.xhand import xhand_x as xhx

from ultralytics import YOLO
from utils.precise_sleep import precise_wait
from wilor.models import WiLoR, load_wilor
from wilor.utils.renderer import Renderer, cam_crop_to_full

WIDTH = 1280
HEIGHT = 720


def wilor_to_wrs(queue1: multiprocessing.Queue):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, model_cfg = load_wilor(checkpoint_path='./pretrained_models/wilor_final.ckpt',
                                  cfg_path='./pretrained_models/model_config.yaml')

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)  ##3Dグラフィックを表示するためのオブジェクト
    model = model.to(device)
    model.eval()

    detector = YOLO(f'./pretrained_models/detector.pt').to(device)
    hand_detector = HandDetector_multifinger(device, model, model_cfg, renderer, detector, hand_type="Right",
                                             detect_hand_num=1, WIDTH=1280, HEIGHT=720)

    # RealSenseカメラの設定
    pipeline_hand = rs.pipeline()
    config_hand = rs.config()
    # config_hand.enable_device('243122072240')

    # カメラのストリームを設定（RGBと深度）
    config_hand.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)  # 30は毎秒フレーム数
    config_hand.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

    # spatial_filterのパラメータ
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 1)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    # hole_filling_filterのパラメータ
    hole_filling = rs.hole_filling_filter()
    # disparity
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    ##depthとcolorの画角がずれないようにalignを生成
    align = rs.align(rs.stream.color)
    # パイプラインを開始
    pipeline_hand.start(config_hand)

    try:
        while True:
            frames = pipeline_hand.wait_for_frames()
            aligned_frames1 = align.process(frames)
            color_frame = aligned_frames1.get_color_frame()
            if not color_frame:
                continue

            # カラーカメラの内部パラメータを取得
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            ##深度フレームの取得
            depth_frame = aligned_frames1.get_depth_frame()
            if not depth_frame:
                continue

            # フィルタ処理
            filter_frame = spatial.process(depth_frame)
            filter_frame = disparity_to_depth.process(filter_frame)
            filter_frame = hole_filling.process(filter_frame)
            result_depth_frame = filter_frame.as_depth_frame()

            if not result_depth_frame:
                continue

            # BGR画像をNumPy配列に変換
            frame = np.asanyarray(color_frame.get_data())

            img_vis, detected_hand_count, reconstructions = hand_detector.run_wilow_model(frame, IoU_threshold=0.3)
            detected_time = time.time()
            output_img, text = hand_detector.render_reconstruction(img_vis, detected_hand_count, reconstructions)

            # 結果を表示
            cv2.imshow('Hand Tracking', output_img)

            if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
                break

            if detected_hand_count == 1:
                # q=reconstructions['joints_points'][0]
                q = [reconstructions['joints_points'][0], reconstructions['rotmat'][0]]
            else:
                q = None

            queue1.put(q)

    finally:
        # パイプラインを停止
        pipeline_hand.stop()
        cv2.destroyAllWindows()


def wrs(queue1: multiprocessing.Queue):
    base = wd.World(cam_pos=rm.vec(0.5, 0.5, 0.7), lookat_pos=rm.vec(0, 0, .0))
    mgm.gen_frame(pos=[0, 0, 0], rotmat=rm.eye(3)).attach_to(base)
    xhand = xhr.XHandRight(pos=rm.vec(0, 0, 0), rotmat=rm.rotmat_from_euler(0, 0, 0))
    xhexe = xhx.XHandX("/dev/ttyUSB0")
    xhand_k = xhand_K()

    t_start = time.monotonic()
    iter_idx = 0
    command_latency = 0.1
    dt = 0.1

    recording = False
    is_recorded = False
    last_record = False
    jnt_list = None

    onscreen_list = []
    meshscreen_list = []
    thumb_list = []
    index_list = []
    middle_list = []
    ring_list = []
    pinky_list = []
    current_joints_list = None
    value = 0

    def update(iter_idx, onscreen_list, thumb_list, index_list, middle_list, ring_list, pinky_list, current_joints_list,
               task):
        t1 = time.time()

        # calculate timing
        t_cycle_end = t_start + (iter_idx + 1) * dt  ##indexが終わるまでの時間
        t_sample = t_cycle_end - command_latency
        t_command_target = t_cycle_end + dt

        q = queue1.get(timeout=1000)

        # precise_wait(t_sample)
        if q is not None:

            rotmat1 = np.array(
                [[1, 0, 0], [0, np.cos(-np.pi / 2), -np.sin(-np.pi / 2)], [0, np.sin(-np.pi / 2), np.cos(-np.pi / 2)]])
            rotmat2 = np.array(
                [[np.cos(-np.pi / 2), 0, np.sin(-np.pi / 2)], [0, 1, 0], [-np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]])
            rotmat3 = np.array(
                [[np.cos(-np.pi / 2), 0, -np.sin(-np.pi / 2)], [0, 1, 0], [np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]])
            rotmat4 = np.array([[1, 0, 0], [0, np.cos(np.pi), -np.sin(np.pi)], [0, np.sin(np.pi), np.cos(np.pi)]])

            desired_joints_angles = xhand_k.fingertip_ik_mapping_xhand(human_hand_keypoints3d=q[0],
                                                                       human_hand_orientation=q[1],
                                                                       seed_angles=current_joints_list)

            xhand.goto_given_conf(desired_joints_angles)
            received_data=xhexe.goto_given_conf_and_get_hand_state(desired_joints_angles)

            max_sensor_value=min([min(i) for i in received_data['sensor_data'][1].force_data])
            print(f"max:{max_sensor_value }")




            print(f"received_data:{received_data['finger_states'][4].position}")

            for ele in onscreen_list:
                ele.detach()
            for mesh in meshscreen_list:
                mesh.detach()
            for tlist in thumb_list:
                tlist.detach()
            for ilist in index_list:
                ilist.detach()
            for mlist in middle_list:
                mlist.detach()
            for rlist in ring_list:
                rlist.detach()
            for plist in pinky_list:
                plist.detach()

            fk_thumb_position, fk_thumb_rot = xhand_k.finger_forward_kinematics('thumb', desired_joints_angles[:3])
            fk_index_position, fk_index_rot = xhand_k.finger_forward_kinematics('index', desired_joints_angles[3:6])
            fk_middle_position, fk_middle_rot = xhand_k.finger_forward_kinematics('middle', desired_joints_angles[6:8])
            fk_ring_position, fk_ring_rot = xhand_k.finger_forward_kinematics('ring', desired_joints_angles[8:10])
            fk_pinky_position, fk_pinky_rot = xhand_k.finger_forward_kinematics('pinky', desired_joints_angles[10:12])

            onscreen_list.append(xhand.gen_meshmodel())
            onscreen_list[-1].attach_to(base)
            meshscreen_list.append(mgm.gen_frame(pos=[0, 0, 0], rotmat=rotmat1 @ q[1] @ rotmat2))
            meshscreen_list[-1].attach_to(base)
            # thumb_list.append(mgm.gen_frame(pos=(rotmat4 @ fk_thumb_position.T).T, rotmat=rotmat4 @ fk_thumb_rot))
            # thumb_list[-1].attach_to(base)
            # index_list.append(mgm.gen_frame(pos=(rotmat4 @ fk_index_position.T).T, rotmat=rotmat4 @ fk_index_rot))
            # index_list[-1].attach_to(base)
            # middle_list.append(mgm.gen_frame(pos=(rotmat4 @ fk_middle_position.T).T, rotmat=rotmat4 @ fk_middle_rot))
            # middle_list[-1].attach_to(base)
            # ring_list.append(mgm.gen_frame(pos=(rotmat4 @ fk_ring_position.T).T, rotmat=rotmat4 @ fk_ring_rot))
            # ring_list[-1].attach_to(base)
            # pinky_list.append(mgm.gen_frame(pos=(rotmat4 @ fk_pinky_position.T).T, rotmat=rotmat4 @ fk_pinky_rot))
            # pinky_list[-1].attach_to(base)

            current_joints_list = desired_joints_angles

            # xhexe.goto_given_conf(rm.np.array(list(angles.values())))

        precise_wait(t_cycle_end)
        iter_idx += 1
        t2 = time.time()

        # print(f"周期:{t2-t1}")
        return task.cont

    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[iter_idx, onscreen_list, thumb_list, index_list, middle_list, ring_list,
                                     pinky_list, current_joints_list], appendTask=True)
    base.run()
    # cv2.destroyAllWindows()

    # def update(iter_idx, onscreen_list, thumb_list, index_list, middle_list, ring_list, pinky_list, current_joints_list,
    #            task):
    #     t1 = time.time()
    #
    #     # calculate timing
    #     t_cycle_end = t_start + (iter_idx + 1) * dt  ##indexが終わるまでの時間
    #     t_sample = t_cycle_end - command_latency
    #     t_command_target = t_cycle_end + dt
    #
    #     q = queue1.get(timeout=1000)
    #
    #     # precise_wait(t_sample)
    #     if q is not None:
    #
    #         rotmat1 = np.array(
    #             [[1, 0, 0], [0, np.cos(-np.pi / 2), -np.sin(-np.pi / 2)], [0, np.sin(-np.pi / 2), np.cos(-np.pi / 2)]])
    #         rotmat2 = np.array(
    #             [[np.cos(-np.pi / 2), 0, np.sin(-np.pi / 2)], [0, 1, 0], [-np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]])
    #         rotmat3 = np.array(
    #             [[np.cos(-np.pi / 2), 0, -np.sin(-np.pi / 2)], [0, 1, 0], [np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]])
    #         rotmat4 = np.array([[1, 0, 0], [0, np.cos(np.pi), -np.sin(np.pi)], [0, np.sin(np.pi), np.cos(np.pi)]])
    #
    #         desired_joints_angles = xhand_k.fingertip_ik_mapping_xhand(human_hand_keypoints3d=q[0],
    #                                                                    human_hand_orientation=q[1],
    #                                                                    seed_angles=current_joints_list)
    #
    #         # xhand.goto_given_conf(desired_joints_angles)
    #         for ele in onscreen_list:
    #             ele.detach()
    #         for mesh in meshscreen_list:
    #             mesh.detach()
    #         for tlist in thumb_list:
    #             tlist.detach()
    #         for ilist in index_list:
    #             ilist.detach()
    #         for mlist in middle_list:
    #             mlist.detach()
    #         for rlist in ring_list:
    #             rlist.detach()
    #         for plist in pinky_list:
    #             plist.detach()
    #
    #         xhand_thumb_position = q[0][4] - q[0][0]
    #         xhand_index_position = q[0][8] - q[0][0]
    #         xhand_middle_position = q[0][12] - q[0][0]
    #         xhand_ring_position = q[0][16] - q[0][0]
    #         xhand_pinky_position = q[0][20] - q[0][0]
    #
    #         xhand_thumb_position = (q[1].T @ xhand_thumb_position.T).T + xhand_k.wrist_offset
    #         xhand_index_position = (q[1].T @ xhand_index_position.T).T + xhand_k.wrist_offset
    #         xhand_middle_position = (q[1].T @ xhand_middle_position.T).T + xhand_k.wrist_offset
    #         xhand_ring_position = (q[1].T @ xhand_ring_position.T).T + xhand_k.wrist_offset
    #         xhand_pinky_position = (q[1].T @ xhand_pinky_position.T).T + xhand_k.wrist_offset
    #
    #         xhand_thumb_position = (xhand_k.human_ori_to_wrs @ xhand_thumb_position.T).T
    #         xhand_index_position = (xhand_k.human_ori_to_wrs @ xhand_index_position.T).T
    #         xhand_middle_position = (xhand_k.human_ori_to_wrs @ xhand_middle_position.T).T
    #         xhand_ring_position = (xhand_k.human_ori_to_wrs @ xhand_ring_position.T).T
    #         xhand_pinky_position = (xhand_k.human_ori_to_wrs @ xhand_pinky_position.T).T
    #
    #         fk_thumb_position, fk_thumb_rot = xhand_k.finger_forward_kinematics('thumb', desired_joints_angles[:3])
    #         fk_index_position, fk_index_rot = xhand_k.finger_forward_kinematics('index', desired_joints_angles[3:6])
    #         fk_middle_position, fk_middle_rot = xhand_k.finger_forward_kinematics('middle', desired_joints_angles[6:8])
    #         fk_ring_position, fk_ring_rot = xhand_k.finger_forward_kinematics('ring', desired_joints_angles[8:10])
    #         fk_pinky_position, fk_pinky_rot = xhand_k.finger_forward_kinematics('pinky', desired_joints_angles[10:12])
    #
    #         onscreen_list.append(xhand.gen_meshmodel())
    #         onscreen_list[-1].attach_to(base)
    #         meshscreen_list.append(mgm.gen_frame(pos=[0, 0, 0], rotmat=rotmat1 @ q[1] @ rotmat2))
    #         meshscreen_list[-1].attach_to(base)
    #         # thumb_list.append(mgm.gen_frame(pos=(rotmat4 @ fk_thumb_position.T).T, rotmat=rotmat4 @ fk_thumb_rot))
    #         # thumb_list[-1].attach_to(base)
    #         # index_list.append(mgm.gen_frame(pos=(rotmat4 @ fk_index_position.T).T, rotmat=rotmat4 @ fk_index_rot))
    #         # index_list[-1].attach_to(base)
    #         # middle_list.append(mgm.gen_frame(pos=(rotmat4 @ fk_middle_position.T).T, rotmat=rotmat4 @ fk_middle_rot))
    #         # middle_list[-1].attach_to(base)
    #         # ring_list.append(mgm.gen_frame(pos=(rotmat4 @ fk_ring_position.T).T, rotmat=rotmat4 @ fk_ring_rot))
    #         # ring_list[-1].attach_to(base)
    #         # pinky_list.append(mgm.gen_frame(pos=(rotmat4 @ fk_pinky_position.T).T, rotmat=rotmat4 @ fk_pinky_rot))
    #         # pinky_list[-1].attach_to(base)
    #
    #         # thumb_list.append(mgm.gen_frame(pos=(rotmat4 @ xhand_thumb_position.T).T, rotmat=rotmat4 @ fk_thumb_rot))
    #         # thumb_list[-1].attach_to(base)
    #         # index_list.append(mgm.gen_frame(pos=(rotmat4 @ xhand_index_position.T).T, rotmat=rotmat4 @ fk_index_rot))
    #         # index_list[-1].attach_to(base)
    #         # middle_list.append(mgm.gen_frame(pos=(rotmat4 @ xhand_middle_position.T).T, rotmat=rotmat4 @ fk_middle_rot))
    #         # middle_list[-1].attach_to(base)
    #         # ring_list.append(mgm.gen_frame(pos=(rotmat4 @ xhand_ring_position.T).T, rotmat=rotmat4 @ fk_ring_rot))
    #         # ring_list[-1].attach_to(base)
    #         # pinky_list.append(mgm.gen_frame(pos=(rotmat4 @ xhand_pinky_position.T).T, rotmat=rotmat4 @ fk_pinky_rot))
    #         # pinky_list[-1].attach_to(base)
    #
    #         current_joints_list = desired_joints_angles
    #
    #         # xhexe.goto_given_conf(rm.np.array(list(angles.values())))
    #
    #     precise_wait(t_cycle_end)
    #     iter_idx += 1
    #     t2 = time.time()
    #
    #     # print(f"周期:{t2-t1}")
    #     return task.cont
    #
    # taskMgr.doMethodLater(0.01, update, "update",
    #                       extraArgs=[iter_idx, onscreen_list, thumb_list, index_list, middle_list, ring_list,
    #                                  pinky_list,current_joints_list], appendTask=True)
    # base.run()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    queue1 = multiprocessing.Queue(maxsize=1)

    process1 = multiprocessing.Process(target=wilor_to_wrs, args=(queue1,))
    process2 = multiprocessing.Process(target=wrs, args=(queue1,))

    process1.start()
    process2.start()

    process1.join()
    process2.join()

    print("done")
