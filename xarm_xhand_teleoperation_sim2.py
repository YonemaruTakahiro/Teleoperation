import hand_keypoints_transform
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
from utils.data_class import multi_finger_animation, WiLor_Data, Data
from xhand_class_ikpy import xhand_K
from utils.teleoperation_utils import abnormal_jnts_change_detection

import pyrealsense2 as rs
import multiprocessing
import torch
import time
import cv2
import numpy as np
from queue import Empty
from loguru import logger
from wrs import wd, rm, ur3d, rrtc, mgm, mcm
from wrs.robot_sim.robots.xarm7_dual import xarm7_dual as x7xh

WIDTH = 1280
HEIGHT = 720

def wilor_to_xhand(queue1: multiprocessing.Queue):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaa")
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
            # 1つ目のフレームを取得
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

            outputs=pipe.predict(frame)

            detected_time = time.time()
            # output_img, text = hand_detector.render_reconstruction(img_vis, detected_hand_count, reconstructions)

            # 結果を表示
            cv2.imshow('Hand Tracking', frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
                break

            print(f"outputs:{outputs}")
            if len(outputs) > 0:
                for hand in outputs:
                    if hand["is_right"]:
                        print("AAAAAAAAA")
                        keypoints = hand["wilor_preds"]["pred_keypoints_3d"]  # Shape: (N, 21, 3)  -> 3D Keypoints
                        kpts_2d = hand["wilor_preds"]["pred_keypoints_2d"]
                        eef_pos = hand_keypoints_transform.calib(kpts_2d[0].tolist(), result_depth_frame, intrinsics)
                        eef_pos = hand_keypoints_transform.coor_trans_from_rs_to_xarm(eef_pos)
                        human_hand_rotmat = hand["wilor_preds"]["global_orient"]
                        eef_rotmat = hand_keypoints_transform.rotmat_to_xarm(human_hand_rotmat)
                        # rotmat = hand_keypoints_transform.fixed_rotmat_to_wrs(keypoints['joints'][0])
                        q = [eef_pos, eef_rotmat, keypoints[0], human_hand_rotmat]
                    break
            else:
                q = None

            queue1.put(q)

    finally:
        # パイプラインを停止
        pipeline_hand.stop()
        cv2.destroyAllWindows()


def wrs(queue1: multiprocessing.Queue):
    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)

    robot = x7xh.XArm7Dual(enable_cc=True)
    xhand_k = xhand_K()

    start_manipulator_conf = np.radians(np.array([30, 40, 0, 90, -90, 0, -180]))
    start_manipulator_pos, start_manipulator_rotmat = robot.fk(start_manipulator_conf, toggle_jacobian=False)

    start_xhand_jnts_values = np.array([0] * 12)

    robot.goto_given_conf(jnt_values=start_manipulator_conf)
    # robot.cc.show_cdprim()
    start_robot_model = robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True, toggle_jnt_frames=False)
    start_mesh_model = mgm.gen_frame(pos=start_manipulator_pos, rotmat=start_manipulator_rotmat)

    # start_mesh_model.attach_to(base)
    start_robot_model.attach_to(base)

    # robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)

    animation_data = multi_finger_animation(start_manipulator_pos, start_manipulator_rotmat, start_manipulator_conf,
                                            start_xhand_jnts_values, start_mesh_model, start_robot_model)

    wilor_data = WiLor_Data()

    def update(animation_data, wilor_data, task):

        q = queue1.get(timeout=20)

        if q is not None:
            if q[0] is not None and q[1] is not None and q[2] is not None and q[3] is not None:
                wilor_data.eef_pos = q[0]
                wilor_data.eef_rotmat = q[1]
                wilor_data.keypoints_3d = q[2]
                wilor_data.human_hand_rotmat = q[3]

                if animation_data.pos_error(wilor_data.eef_pos,
                                            animation_data.current_pos) > 0.0015 or animation_data.rotmat_error(
                        wilor_data.eef_rotmat, animation_data.current_rotmat) > 0.05:
                    animation_data.tgt_pos = animation_data.current_pos
                    animation_data.tgt_rotmat = animation_data.current_rotmat
                else:
                    animation_data.tgt_pos = wilor_data.eef_pos
                    animation_data.tgt_rotmat = wilor_data.eef_rotmat

                manipulator_jnt_values = robot.realtime_ik(animation_data.tgt_pos, animation_data.tgt_rotmat,
                                                           seed_jnt_values=animation_data.current_manipulator_jnt_values,
                                                           toggle_dbg=False)

                if manipulator_jnt_values is None:
                    print("No IK solution found!")
                    animation_data.next_manipulator_jnt_values = animation_data.current_manipulator_jnt_values
                else:
                    if abnormal_jnts_change_detection(animation_data.current_manipulator_jnt_values,
                                                      manipulator_jnt_values):
                        animation_data.next_manipulator_jnt_values = animation_data.current_manipulator_jnt_values
                    else:
                        animation_data.next_manipulator_jnt_values = manipulator_jnt_values

                animation_data.next_xhand_jnts_values = xhand_k.fingertip_ik_mapping_xhand(
                    human_hand_keypoints3d=wilor_data.keypoints_3d,
                    human_hand_orientation=wilor_data.human_hand_rotmat,
                    seed_angles=animation_data.current_xhand_jnt_values)

                animation_data.mesh_model.detach()
                animation_data.robot_model.detach()

                robot.goto_given_conf(jnt_values=animation_data.next_manipulator_jnt_values)
                robot.end_effector.goto_given_conf(animation_data.next_xhand_jnts_values)

                animation_data.robot_model = robot.gen_meshmodel(toggle_tcp_frame=True)
                animation_data.mesh_model = mgm.gen_frame(pos=animation_data.tgt_pos, rotmat=animation_data.tgt_rotmat)

                animation_data.mesh_model.attach_to(base)
                animation_data.robot_model.attach_to(base)

                animation_data.current_xhand_jnt_values = animation_data.next_xhand_jnts_values
                animation_data.current_manipulator_jnt_values = animation_data.next_manipulator_jnt_values

        animation_data.count += 1

        return task.again

    taskMgr.doMethodLater(0.0, update, "update",
                          extraArgs=[animation_data, wilor_data],
                          appendTask=True)

    base.run()


if __name__ == "__main__":
    queue1 = multiprocessing.Queue(maxsize=1)
    # queue2 = multiprocessing.Queue(maxsize=1)
    # queue3 = multiprocessing.Queue(maxsize=1)

    process1 = multiprocessing.Process(target=wilor_to_xhand, args=(queue1,))
    # process2 = multiprocessing.Process(target=get_frame_robot, args=(queue2,))
    # process3 = multiprocessing.Process(target=get_frame_robot_wrist, args=(queue3,))
    process4 = multiprocessing.Process(target=wrs, args=(queue1,))

    process1.start()
    # process2.start()
    # process3.start()
    process4.start()

    process1.join()
    # process2.join()
    # process3.join()
    process4.join()

    print("done")
