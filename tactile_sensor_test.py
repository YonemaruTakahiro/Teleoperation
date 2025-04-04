import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from wrs.robot_con.xhand import xhand_x as xhx


fig, ax = plt.subplots()

# RGB画像の初期化
rgb_image = np.zeros((12, 10, 3))  # 12行10列のRGB画像



hand = xhx.XHandX(port="/dev/ttyUSB0", baudrate=3000000)
hand.get_version()
while True:
    hand_state=hand.goto_given_conf_and_get_hand_state([0.5]*12)

    tactile_value=np.array(hand_state['sensor_data'][0].force_data)

    norm = Normalize(vmin=-20, vmax=50)

    # RGBに変換
    r_values = norm(tactile_value[:, 0])
    g_values = norm(tactile_value[:, 1])
    b_values = norm(tactile_value[:, 2])

    # rgb_image = np.zeros((12, 10, 3))  # 12行10列のRGB画像を作成 (120個の点に対応)
    for i, (r, g, b) in enumerate(zip(r_values, g_values, b_values)):
        row=11-i % 12
        col=i // 12
        rgb_image[row, col] = [r, g, b]

    im = ax.imshow(rgb_image)
    ax.axis('off')  # 軸を非表示に

    # 画像の更新
    plt.pause(.01)



# Close connection
hand.close()




# # 仮のセンサデータ (120個の点、それぞれx, y, zの値をランダムに生成)
# num_points = 120
# points = np.random.rand(num_points, 3)  # 120x3 の配列 (x, y, z)
#
# # x, y, zの値をそれぞれRGBにマッピングする
# # ここでは簡単のため、x, y, zをそれぞれR, G, Bに直接マッピングします
# # 正規化を行って0-1の範囲に収める
# norm = Normalize(vmin=0, vmax=1)
#
# # RGBに変換
# r_values = norm(points[:, 0])
# g_values = norm(points[:, 1])
# b_values = norm(points[:, 2])
#
# # RGBカラーの画像データを作成
# # ここでは、120個の点を横一列に並べた画像にしています。
# rgb_image = np.zeros((10, 12, 3))  # 10行12列のRGB画像を作成 (120個の点に対応)
# for i, (r, g, b) in enumerate(zip(r_values, g_values, b_values)):
#     row = i // 12
#     col = i % 12
#     rgb_image[row, col] = [r, g, b]
#
# # 画像表示
# plt.imshow(rgb_image)
# plt.axis('off')  # 軸を非表示に
# plt.show()