import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from enum import IntEnum

def get_profiles():
    ctx = rs.context()
    devices = ctx.query_devices()

    color_profiles = []
    depth_profiles = []
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print('Sensor: {}, {}'.format(name, serial))
        print('Supported video formats:')
        for sensor in device.query_sensors():
            for stream_profile in sensor.get_stream_profiles():
                stream_type = str(stream_profile.stream_type())

                if stream_type in ['stream.color', 'stream.depth']:
                    v_profile = stream_profile.as_video_stream_profile()
                    fmt = stream_profile.format()
                    w, h = v_profile.width(), v_profile.height()
                    fps = v_profile.fps()

                    video_type = stream_type.split('.')[-1]
                    print('  {}: width={}, height={}, fps={}, fmt={}'.format(
                        video_type, w, h, fps, fmt))
                    if video_type == 'color':
                        color_profiles.append((w, h, fps, fmt))
                    else:
                        depth_profiles.append((w, h, fps, fmt))

    return color_profiles, depth_profiles


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


class RealsesneProcessor:
    def __init__(
        self,
        first_D435_serial,
        second_D435_serial,
        total_frame,
        store_frame=False,
        out_directory=None,
        save_hand=False,
        enable_visualization=True,
    ):
        self.first_D435_serial = first_D435_serial
        self.second_D435_serial = second_D435_serial
        self.store_frame = store_frame
        self.out_directory = out_directory
        self.total_frame = total_frame
        self.save_hand = save_hand
        self.enable_visualization = enable_visualization
        self.rds = None

        self.color_buffer = []
        self.depth_buffer = []

        self.pose_buffer = []
        self.pose2_buffer = []
        self.pose3_buffer = []

        self.pose2_image_buffer = []
        self.pose3_image_buffer = []

        self.rightHandJoint_buffer = []
        self.leftHandJoint_buffer = []
        self.rightHandJointOri_buffer = []
        self.leftHandJointOri_buffer = []
    
    def get_rs_D435_config(self, D435_serial, D435_pipeline):
        D435_config = rs.config()
        D435_config.enable_device(D435_serial)
        D435_config.enable_stream(rs.stream.pose)

        return D435_config
        
    def configure_stream(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        color_profiles, depth_profiles = get_profiles()
        w, h, fps, fmt = depth_profiles[1]
        config.enable_stream(rs.stream.depth, w, h, fmt, fps)
        w, h, fps, fmt = color_profiles[18]
        config.enable_stream(rs.stream.color, w, h, fmt, fps)

        # Configure the D435 1 stream
        ctx = rs.context()#デバイスの検出
        self.D435_pipeline = rs.pipeline(ctx)
        D435_config = rs.config()
        D435_config.enable_device(self.first_D435_serial)

        # Configure the D435 2 stream
        ctx_2 = rs.context()#デバイスの検出
        self.D435_pipeline_2 = rs.pipeline(ctx_2)
        D435_config_2 = self.get_rs_D435_config(
            self.second_D435_serial, self.D435_pipeline_2
        )

        self.D435_pipeline.start(D435_config)
        self.D435_pipeline_2.start(D435_config_2)

        pipeline_profile = self.pipeline.start(config)
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)
        self.depth_scale = depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.vis = None
        if self.enable_visualization:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.vis.get_view_control().change_field_of_view(step=1.0)

    def get_rgbd_frame_from_realsense(self, enable_visualization=False):
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = (
            np.asanyarray(aligned_depth_frame.get_data()) // 4
        )  # L515 camera need to divide by 4 to get metric in meter
        color_image = np.asanyarray(color_frame.get_data())

        rgbd = None
        if enable_visualization:
            depth_image_o3d = o3d.geometry.Image(depth_image)
            color_image_o3d = o3d.geometry.Image(color_image)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image_o3d,
                depth_image_o3d,
                depth_trunc=4.0,
                convert_rgb_to_intensity=False,
            )
        return rgbd, depth_image, color_image