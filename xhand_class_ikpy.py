from ikpy import chain
import numpy as np
import yaml


def get_yaml_data(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


class xhand_K():
    def __init__(self):
        urdf_path = "xhand_model/xhand_right_test.urdf"
        # Loading xhand configs
        self.hand_configs = get_yaml_data("xhand_model/configs/xhand_info.yaml")
        self.finger_configs = get_yaml_data("xhand_model/configs/xhand_link_info.yaml")

        self.wrist_offset = [-0.02, -0.02, 0.03]#wilorの座標系
        self.human_ori_to_wrs = np.array(
            [[1, 0, 0], [0, np.cos(np.pi), -np.sin(np.pi)], [0, np.sin(np.pi), np.cos(np.pi)]]) @ np.array(
            [[np.cos(-np.pi / 2), 0, -np.sin(-np.pi / 2)], [0, 1, 0], [np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]])

        # Parsing chains from the urdf file
        self.chains = {}
        for finger in self.hand_configs['fingers'].keys():
            self.chains[finger] = chain.Chain.from_urdf_file(
                urdf_path,
                base_elements=[self.finger_configs['links_info']['base']['link'],
                               self.finger_configs['links_info'][finger]['link']],
                active_links_mask=[False] + [True] * (self.hand_configs['fingers'][finger]['joints_per_finger'] + 1),
                name=finger
            )

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

    def finger_forward_kinematics(self, finger_type, input_angles):
        # Checking if the number of angles is equal to len(input_angles)
        if len(input_angles) != self.hand_configs['fingers'][finger_type]['joints_per_finger']:
            print(f'{finger_type}:Incorrect number of angles')
            return

            # Checking if the input finger type is a valid one
        if finger_type not in self.hand_configs['fingers'].keys():
            print('Finger type does not exist')
            return

        # Clipping the input angles based on the finger type
        finger_info = self.finger_configs['links_info'][finger_type]
        for iterator in range(len(input_angles)):
            if input_angles[iterator] > finger_info['joint_max'][iterator]:
                input_angles[iterator] = finger_info['joint_max'][iterator]
            elif input_angles[iterator] < finger_info['joint_min'][iterator]:
                input_angles[iterator] = finger_info['joint_min'][iterator]

        # Padding values at the beginning and the end to get for a (1x6) array
        input_angles = list(input_angles)
        input_angles.insert(0, 0)
        input_angles.append(0)

        # Performing Forward Kinematics
        output_frame = self.chains[finger_type].forward_kinematics(input_angles)
        return output_frame[:3, 3], output_frame[:3, :3]

    def finger_inverse_kinematics(self, finger_type, input_position, seed=None):
        # Checking if the input figner type is a valid one
        if finger_type not in self.hand_configs['fingers'].keys():
            print('Finger type does not exist')
            return

        if seed is not None:
            # Checking if the number of angles is equal to 4
            if len(seed) != self.hand_configs['fingers'][finger_type]['joints_per_finger']:
                print('Incorrect seed array length')
                return

                # Clipping the input angles based on the finger type
            finger_info = self.finger_configs['links_info'][finger_type]
            for iterator in range(len(seed)):
                if seed[iterator] > finger_info['joint_max'][iterator]:
                    seed[iterator] = finger_info['joint_max'][iterator]
                elif seed[iterator] < finger_info['joint_min'][iterator]:
                    seed[iterator] = finger_info['joint_min'][iterator]

            # Padding values at the beginning and the end to get for a (1x6) array
            seed = list(seed)
            seed.insert(0, 0)
            seed.append(0)
        print(f"{finger_type}:{seed}")
        output_angles = self.chains[finger_type].inverse_kinematics(input_position, initial_position=seed)

        return output_angles[1:-1]

    def fingertip_ik_mapping_xhand(self, human_hand_keypoints3d, human_hand_orientation, seed_angles=None):
        # wrsのxhandの向きに変換
        rotmat1 = np.array(
            [[1, 0, 0], [0, np.cos(-np.pi / 2), -np.sin(-np.pi / 2)], [0, np.sin(-np.pi / 2), np.cos(-np.pi / 2)]])
        rotmat2 = np.array(
            [[np.cos(-np.pi / 2), 0, np.sin(-np.pi / 2)], [0, 1, 0], [-np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]])

        xhand_thumb_position = human_hand_keypoints3d[4] - human_hand_keypoints3d[0]
        xhand_index_position = human_hand_keypoints3d[8] - human_hand_keypoints3d[0]
        xhand_middle_position = human_hand_keypoints3d[12] - human_hand_keypoints3d[0]
        xhand_ring_position = human_hand_keypoints3d[16] - human_hand_keypoints3d[0]
        xhand_pinky_position = human_hand_keypoints3d[20] - human_hand_keypoints3d[0]

        xhand_thumb_position = (human_hand_orientation.T @ xhand_thumb_position.T).T + self.wrist_offset
        xhand_index_position = (human_hand_orientation.T @ xhand_index_position.T).T + self.wrist_offset
        xhand_middle_position = (human_hand_orientation.T @ xhand_middle_position.T).T + self.wrist_offset
        xhand_ring_position = (human_hand_orientation.T @ xhand_ring_position.T).T + self.wrist_offset
        xhand_pinky_position = (human_hand_orientation.T @ xhand_pinky_position.T).T + self.wrist_offset

        xhand_thumb_position = (self.human_ori_to_wrs @ xhand_thumb_position.T).T
        xhand_index_position = (self.human_ori_to_wrs @ xhand_index_position.T).T
        xhand_middle_position = (self.human_ori_to_wrs @ xhand_middle_position.T).T
        xhand_ring_position = (self.human_ori_to_wrs @ xhand_ring_position.T).T
        xhand_pinky_position = (self.human_ori_to_wrs @ xhand_pinky_position.T).T

        if seed_angles is None:
            angles = self.compute_selected_angles(human_hand_keypoints3d)

            angles_list = []
            for angle in angles.values():
                angles_list.append(angle)

            thumb_joints_angles = self.finger_inverse_kinematics('thumb', xhand_thumb_position, seed=angles_list[:3])
            index_joints_angles = self.finger_inverse_kinematics('index', xhand_index_position, seed=angles_list[3:6])
            middle_joints_angles = self.finger_inverse_kinematics('middle', xhand_middle_position,
                                                                  seed=angles_list[6:8])
            ring_joints_angles = self.finger_inverse_kinematics('ring', xhand_ring_position, seed=angles_list[8:10])
            pinky_joints_angles = self.finger_inverse_kinematics('pinky', xhand_pinky_position, seed=angles_list[10:12])

        else:
            thumb_joints_angles = self.finger_inverse_kinematics('thumb', xhand_thumb_position, seed=seed_angles[:3])
            index_joints_angles = self.finger_inverse_kinematics('index', xhand_index_position, seed=seed_angles[3:6])
            middle_joints_angles = self.finger_inverse_kinematics('middle', xhand_middle_position,
                                                                  seed=seed_angles[6:8])
            ring_joints_angles = self.finger_inverse_kinematics('ring', xhand_ring_position, seed=seed_angles[8:10])
            pinky_joints_angles = self.finger_inverse_kinematics('pinky', xhand_pinky_position, seed=seed_angles[10:12])

        desired_joints_angles = np.concatenate(
            [thumb_joints_angles, index_joints_angles, middle_joints_angles, ring_joints_angles, pinky_joints_angles],
            0)

        return desired_joints_angles
