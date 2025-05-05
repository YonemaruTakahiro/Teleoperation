import pyrealsense2 as rs
import numpy as np
import cv2

# パイプラインの構築
pipeline = rs.pipeline()
config = rs.config()

# ストリームの設定（カラーと深度）
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# ストリーミング開始
pipeline.start(config)

try:
    while True:
        # フレームを待機して取得
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # NumPy配列に変換
        color_image = np.asanyarray(color_frame.get_data())

        # 画像表示
        cv2.imshow('RealSense Color Stream', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 終了処理
    pipeline.stop()
    cv2.destroyAllWindows()
