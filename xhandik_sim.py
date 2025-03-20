from hand_detector_multi_finger import HandDetector_multifinger
# from xhand_class_pybullet import xhandIK
from utils.data_class import animation,handdata,Data
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
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
import struct

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
                q=hand_detector.compute_selected_angles(reconstructions['joints_points'][0])
                # q=[reconstructions['joints_points'][0],reconstructions['rotmat'][0]]
            else:
                q = None

            queue1.put(q)

    finally:
        # パイプラインを停止
        pipeline_hand.stop()
        cv2.destroyAllWindows()



def wrs(queue1: multiprocessing.Queue):

    base = wd.World(cam_pos=rm.vec(0.5, 0.5, 0.7), lookat_pos=rm.vec(0, 0, .0))
    xhand = xhr.XHandRight(pos=rm.vec(0, 0, 0), rotmat=rm.rotmat_from_euler(0, 0, 0))
    # xhexe = xhx.XHandX("/dev/ttyUSB0")
    # xhandik=xhandIK()

    t_start = time.monotonic()
    iter_idx = 0
    command_latency=0.1
    dt=0.1

    recording=False
    is_recorded=False
    last_record=False
    jnt_list = None

    onscreen_list= []
    value = 0
    def update(iter_idx, onscreen_list, task):
        t1 = time.time()

        # calculate timing
        t_cycle_end = t_start + (iter_idx + 1) * dt  ##indexが終わるまでの時間
        t_sample = t_cycle_end - command_latency
        t_command_target = t_cycle_end + dt

        q = queue1.get(timeout=1000)

        # precise_wait(t_sample)
        if q is not None:
            # angles=xhandik.compute_IK(q[0],q[1])
            xhand.goto_given_conf(rm.np.array(list(q.values())))
            for ele in onscreen_list:
                ele.detach()
            onscreen_list.append(xhand.gen_meshmodel())
            onscreen_list[-1].attach_to(base)
            # xhexe.goto_given_conf(rm.np.array(list(angles.values())))
        precise_wait(t_cycle_end)
        iter_idx += 1
        t2=time.time()

        # print(f"周期:{t2-t1}")
        return task.cont

    taskMgr.doMethodLater(0.01, update, "update", extraArgs=[iter_idx, onscreen_list], appendTask=True)
    base.run()
    cv2.destroyAllWindows()

# def wrs(queue1: multiprocessing.Queue):
#
#     # base = wd.World(cam_pos=rm.vec(0.5, 0.5, 0.7), lookat_pos=rm.vec(0, 0, .0))
#     # xhand = xhr.XHandRight(pos=rm.vec(0, 0, 0), rotmat=rm.rotmat_from_euler(0, 0, 0))
#     # xhexe = xhx.XHandX("/dev/ttyUSB0")
#
#     t_start = time.monotonic()
#     iter_idx = 0
#     command_latency = 0.1
#     dt = 0.06
#
#     recording = False
#     is_recorded = False
#     last_record = False
#     jnt_list = None
#
#     # onscreen_list= []
#     value = 0
#     try:
#         while True:
#             t1 = time.time()
#
#             # calculate timing
#             t_cycle_end = t_start + (iter_idx + 1) * dt  ##indexが終わるまでの時間
#             t_sample = t_cycle_end - command_latency
#             t_command_target = t_cycle_end + dt
#
#             angles = queue1.get(timeout=10)
#
#             # precise_wait(t_sample)
#             # if angles is not None:
#                 # for joint, angle in angles.items():
#                 #     print(f"{joint}: {np.degrees(angle):.2f}°")
#                 # xhand.goto_given_conf(rm.np.array(list(angles.values())))
#                 # for ele in onscreen_list:
#                 #     ele.detach()
#                 # onscreen_list.append(xhand.gen_meshmodel())
#                 # onscreen_list[-1].attach_to(base)
#                 # xhexe.goto_given_conf(rm.np.array(list(angles.values())))
#             precise_wait(t_cycle_end)
#             iter_idx += 1
#             t2 = time.time()
#
#             print(f"周期:{t2 - t1}")
#
#     except Empty:
#         logger.error(f"Fail to fetch image from camera in 10 secs. Please check your web camera device.")
#         cv2.destroyAllWindows()
#
#
#     finally:
#         cv2.destroyAllWindows()



if __name__ == "__main__":
    queue1 = multiprocessing.Queue(maxsize=1)

    process1 = multiprocessing.Process(target=wilor_to_wrs, args=(queue1,))
    process2 = multiprocessing.Process(target=wrs, args=(queue1,))

    process1.start()
    process2.start()

    process1.join()
    process2.join()

    print("done")
