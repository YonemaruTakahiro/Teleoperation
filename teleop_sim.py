from hand_detector_wilor import HandDetector_wilor
from data_class import animation_sim,handdata,Data
from ultralytics import YOLO
import pyrealsense2 as rs
import multiprocessing
import torch
from precise_sleep import precise_wait
from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
import time
import cv2
import numpy as np
from queue import Empty
from loguru import logger
from wrs import wd, rm, ur3d, rrtc, mgm, mcm


WIDTH = 1280
HEIGHT = 720

def wilor_to_wrs(queue: multiprocessing.Queue):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, model_cfg = load_wilor(checkpoint_path='./pretrained_models/wilor_final.ckpt',
                                  cfg_path='./pretrained_models/model_config.yaml')
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)  ##3Dグラフィックを表示するためのオブジェクト
    model = model.to(device)
    model.eval()

    detector = YOLO(f'./pretrained_models/detector.pt').to(device)
    hand_detector = HandDetector_wilor(device, model, model_cfg, renderer, detector, hand_type="Right",
                                       detect_hand_num=1, WIDTH=1280, HEIGHT=720)

    # RealSenseカメラの設定
    pipeline_hand = rs.pipeline()
    config_hand = rs.config()
    config_hand.enable_device('243122072240')

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



            img_vis, detected_hand_count, reconstructions = hand_detector.run_wilow_model(frame, IoU_threshold=0.3)
            detected_time = time.time()
            # output_img, text = hand_detector.render_reconstruction(img_vis, detected_hand_count, reconstructions)

            # 結果を表示
            cv2.imshow('Hand Tracking', img_vis)

            if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
                break

            if detected_hand_count == 1:
                kpts_2d = hand_detector.keypoints_2d_on_image(reconstructions['joints'][0], reconstructions['cam_t'][0],
                                                              reconstructions['focal'])
                pos_3d_wrist = hand_detector.calib(kpts_2d[0].tolist(), result_depth_frame, intrinsics)
                index_finger_mcp=np.array(reconstructions['joints'][0][5].tolist())
                if pos_3d_wrist is not None and index_finger_mcp is not None:
                    pos_3d = hand_detector.coor_trans_from_rs_to_wrs(pos_3d_wrist+index_finger_mcp)
                else:
                    pos_3d=None
                jaw_width = hand_detector.distance_between_fingers_normalization(
                    reconstructions['joints'][0][4].tolist(), reconstructions['joints'][0][8].tolist())
                # rotmat = reconstructions['rotmat'][0]
                # rotmat = hand_detector.rotmat_to_wrs(rotmat)
                rotmat = hand_detector.fixed_rotmat_to_wrs(reconstructions['joints'][0])
                q = [pos_3d, rotmat, jaw_width, detected_time]
            else:
                q = None

            queue.put(q)




    finally:
        # パイプラインを停止
        pipeline_hand.stop()
        cv2.destroyAllWindows()

def wrs(queue: multiprocessing.Queue):
    base = wd.World(cam_pos=[3, 2, 3], lookat_pos=[0.5, 0, 1.1])
    mgm.gen_frame().attach_to(base)



    # robot
    robot = ur3d.UR3Dual()
    robot.use_rgt()



    start_conf = robot.get_jnt_values()
    robot.goto_given_conf(jnt_values=start_conf)
    robot_model = robot.gen_meshmodel(toggle_tcp_frame=True)


    start_pos=np.array([.5, -.35, 1.155])
    start_rotmat=rm.rotmat_from_euler(0, np.pi, np.pi)

    robot.ik(start_pos, start_rotmat,
                 seed_jnt_values=start_conf, toggle_dbg=False)

    count = 0
    mesh_model = mgm.gen_frame(pos=start_pos, rotmat=start_rotmat)

    animation_data=animation_sim(start_pos, start_rotmat,start_conf,mesh_model,robot_model)
    hand_data = handdata()

    animation_data.mesh_model.attach_to(base)
    animation_data.robot_model.attach_to(base)
    animation_data.robot_model.detach()
    robot.goto_given_conf(jnt_values=start_conf)
    robot.change_jaw_width(0.085)
    animation_data.robot_model.attach_to(base)



    def update(animation_data, hand_data, task):
        try:
            q= queue.get(timeout=5)
            # if hand_data.jaw_width is not None and hand_data.hand_pos is not None and hand_data.rotmat is not None:
            #     animation_data.tgt_pos=hand_data.hand_pos[9]
            #     if animation_data.tgt_pos[2]<0.78:
            #         animation_data.tgt_pos[2]=0.9
            # if animation_data.count > 0:
            #     animation_data.mesh_model.detach()

            if q is not None:
                if q[0] is not None and q[1] is not None and q[2] is not None and q[3] is not None:
                    hand_data.hand_pos = q[0]
                    hand_data.hand_rotmat = q[1]
                    hand_data.jaw_width = q[2]
                    hand_data.detected_time = q[3]
                    # print(f"hand_data.detected_time:{hand_data.detected_time}")
                    if animation_data.pos_error(hand_data.hand_pos, animation_data.tgt_rotmat) > 0.000:
                        animation_data.tgt_pos = hand_data.hand_pos
                    if animation_data.rotmat_error(hand_data.hand_rotmat, animation_data.tgt_rotmat) > 0.000:
                        animation_data.tgt_rotmat = hand_data.hand_rotmat
                    animation_data.jaw_width = hand_data.jaw_width
                    if animation_data.tgt_pos[2] < 1.155:
                        animation_data.tgt_pos[2] = 1.155

            jnt_values = robot.ik_sim(animation_data.tgt_pos, animation_data.tgt_rotmat,
                                  seed_jnt_values=animation_data.current_jnt_values, toggle_dbg=False)
            if jnt_values is None:
                print("No IK solution found!")
                animation_data.next_jnt_values = animation_data.current_jnt_values
                animation_data.mesh_model.detach()
                animation_data.robot_model.detach()
                robot.goto_given_conf(jnt_values=animation_data.next_jnt_values)
                robot.change_jaw_width((animation_data.jaw_width / 0.16) * 0.085)
                animation_data.robot_model = robot.gen_meshmodel(toggle_tcp_frame=True)
                animation_data.mesh_model.attach_to(base)
                animation_data.robot_model.attach_to(base)
            else:
                animation_data.next_jnt_values = jnt_values


                animation_data.mesh_model.detach()
                animation_data.robot_model.detach()
                animation_data.mesh_model = mgm.gen_frame(pos=animation_data.tgt_pos,
                                                          rotmat=animation_data.tgt_rotmat)
                robot.goto_given_conf(jnt_values=animation_data.next_jnt_values)
                robot.change_jaw_width((animation_data.jaw_width/0.19)*0.085)
                animation_data.robot_model = robot.gen_meshmodel(toggle_tcp_frame=True)

                animation_data.mesh_model.attach_to(base)
                animation_data.robot_model.attach_to(base)
                animation_data.current_jnt_values = animation_data.next_jnt_values


            animation_data.count += 1

        except Empty:
            logger.error(f"Fail to fetch image from camera in 5 secs. Please check your web camera device.")
            return

        return task.again

    taskMgr.doMethodLater(0.0,update, "update",
                          extraArgs=[animation_data,hand_data],
                          appendTask=True)

    base.run()
    """
    while True:

        try:
            hand_data= queue.get(timeout=5)

            if hand_data is not None:
                print(f"hand:{hand_data[2]}")

        except Empty:
            logger.error(f"Fail to fetch image from camera in 5 secs. Please check your web camera device.")
            return
    """




if __name__ == "__main__":
    queue = multiprocessing.Queue(maxsize=1000)
    producer_process = multiprocessing.Process(target=wilor_to_wrs, args=(queue,))
    consumer_process = multiprocessing.Process(target=wrs, args=(queue,))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")



