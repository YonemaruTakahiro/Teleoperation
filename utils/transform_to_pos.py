import traceback
from wrs.basis.robot_math import rotmat_to_euler,rotmat_from_euler

from wrs import wd, rm, ur3d, rrtc, mgm, mcm

import pickle
from pathlib import Path


input_file_path="only_grouping_two_images.pkl"
input_file_path = Path(input_file_path)  # Pathオブジェクトに変換

robot = ur3d.UR3Dual()
robot.use_rgt()

with input_file_path.open("rb") as f1:
    try:
        train_data = pickle.load(f1)
        print("既存のデータを読み込みました。")
        count = 0
        count_true = False
        for value in train_data["dones"]:
            if value is True and count_true is True:
                print("error!!!!!")
            count_true = value
            if value is True:
                count += 1
        print(f"episode_num:{count}")

    except EOFError:
        traceback.print_exc()


output_data = {"observations":{"rgb_robot":list(),
                                      "rgb_robot_wrist":list(),
                                      "state":list(),},
                              "actions":list(),
                              "rewards":list(),
                              "dones":list(),
                              }

output_data["observations"]["rgb_robot"]=train_data["observations"]["rgb_robot"].copy()
output_data["observations"]["rgb_robot_wrist"]=train_data["observations"]["rgb_robot_wrist"].copy()
output_data["dones"]=train_data["dones"].copy()
output_data["rewards"]=train_data["rewards"].copy()


for value in train_data["observations"]["state"]:
    pos, rotmat = robot.fk(value, toggle_jacobian=False)
    angles=rotmat_to_euler(rotmat, order='sxyz')
    reversed_rot_mat=rotmat_from_euler(angles[0], angles[1], angles[2], order='sxyz')


    output_data["observations"]["state"].append([pos[0],pos[1],pos[2],angles[0], angles[1], angles[2]])

for value in train_data["actions"]:
    pos, rotmat = robot.fk(value, toggle_jacobian=False)
    angles = rotmat_to_euler(rotmat, order='sxyz')
    reversed_rot_mat = rotmat_from_euler(angles[0], angles[1], angles[2], order='sxyz')
    # print(f"rotmat :{rotmat} reversed_rot_mat{reversed_rot_mat}")

    output_data["actions"].append([pos[0], pos[1], pos[2], angles[0], angles[1], angles[2]])

output_file_path="only_grouping_two_images-pos_ver.pkl"
output_file_path = Path(output_file_path)  # Pathオブジェクトに変換

print(f"{len(output_data['observations']['rgb_robot'])} {len(output_data['observations']['rgb_robot_wrist'])} {len(output_data['observations']['state'])} {len(output_data['actions'])} {len(output_data['dones'])}")

with output_file_path.open("wb") as f2:
    pickle.dump(output_data, f2)

print(f"データを更新して保存しました")