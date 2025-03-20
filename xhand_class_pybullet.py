import pybullet as p
import pybullet_data
from yourdfpy import URDF
from transforms3d.euler import quat2euler, euler2quat, quat2mat
from transforms3d.quaternions import mat2quat,axangle2quat, qmult

import numpy as np
import copy
import open3d as o3d


##########################################################################33
def rotate_quaternion(a1=0, a2=0, a3=0):
    q_transform = p.getQuaternionFromEuler([a1, a2, a3])

    return q_transform

def rotate_vector_by_quaternion_using_matrix(v, q):
    # Convert the quaternion to a rotation matrix
    q1 = np.array([q[3], q[0], q[1], q[2]])
    rotation_matrix = quat2mat(q1)

    homogenous_vector = np.array([v[0], v[1], v[2]]).T
    rotated_vector_homogeneous = np.dot(homogenous_vector, rotation_matrix.T)

    return rotated_vector_homogeneous
def rotate_quaternion_xyzw(quaternion_xyzw, axis, angle):
    """
    Rotate a quaternion in the "xyzw" format along a specified axis by a given angle in radians.

    Args:
        quaternion_xyzw (np.ndarray): The input quaternion in "xyzw" format [x, y, z, w].
        axis (np.ndarray): The axis of rotation [x, y, z].
        angle (float): The rotation angle in radians.

    Returns:
        np.ndarray: The rotated quaternion in "xyzw" format [x', y', z', w'].
    """
    # Normalize the axis of rotation
    q1 = np.array(
        [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]]
    )
    q2 = axangle2quat(axis, angle)
    q3 = qmult(q2, q1)

    rotated_quaternion_xyzw = np.array([q3[1], q3[2], q3[3], q3[0]])

    return rotated_quaternion_xyzw
##############################################################################################################

class xhandIK():
    def __init__(self):
        # start pybullet

        p.connect(p.DIRECT)

        # load right xhand
        self.xhandId = p.loadURDF(
            "xhand_model/xhand_right.urdf",
            [0.0, 0.0, 0.0],
            rotate_quaternion(0.0, 0.0, 0.0),
        )

        self.xhand_center_offset = [0.18, 0.03, 0.0]  # Since the root of the LEAP hand URDF is not at the palm's root (it is at the root of the index finger), we set an offset to correct the root location
        self.xhandEndEffectorIndex = [4, 8, 12, 16, 20]  # fingertip joint index
        self.fingertip_offset = np.array([0.1, 0.0, -0.08])  # Since the root of the fingertip mesh in URDF is not at the tip (it is at the right lower part of the fingertip mesh), we set an offset to correct the fingertip location
        self.thumb_offset = np.array([0.1, 0.0, -0.06])  # Same reason for the thumb tip

        self.numJoints = p.getNumJoints(self.xhandId)
        self.hand_lower_limits, self.hand_upper_limits, self.hand_joint_ranges = self.get_joint_limits(self.xhandId)  # get the joint limits of LEAP hand
        self.HAND_Q = np.array([np.pi / 6, -np.pi / 6, np.pi / 3,
                                np.pi / 12, np.pi / 3, np.pi / 3,
                                np.pi / 6, np.pi / 6,
                                np.pi / 6, np.pi / 6,
                                np.pi / 6, np.pi / 6])  # To avoid self-collision of LEAP hand, we define a reference pose for null space IK

        # load URDF of left and right hand for generating pointcloud during forward kinematics
        self.urdf_dict = {}
        self.xhand_urdf = URDF.load("xhand_model/xhand_right.urdf")
        self.urdf_dict["right_xhand"] = {
            "scene": self.xhand_urdf.scene,
            "mesh_list": self._load_meshes(self.xhand_urdf.scene),
        }

        # self.create_target_vis()
        # p.setGravity(0, 0, 0)
        # useRealTimeSimulation = 0
        # p.setRealTimeSimulation(useRealTimeSimulation)

    def get_joint_limits(self, robot):
        joint_lower_limits = []
        joint_upper_limits = []
        joint_ranges = []
        for i in range(p.getNumJoints(robot)):
            joint_info = p.getJointInfo(robot, i)
            if joint_info[2] == p.JOINT_FIXED:
                continue
            joint_lower_limits.append(joint_info[8])
            joint_upper_limits.append(joint_info[9])
            joint_ranges.append(joint_info[9] - joint_info[8])
        return joint_lower_limits, joint_upper_limits, joint_ranges

    def _load_meshes(self, scene):
        mesh_list = []
        for name, g in scene.geometry.items():
            # print(f"g:{g}")
            # mesh = g.to_open3d()
            mesh_list.append(g)

        return mesh_list
    #
    # def _update_meshes(self, type):
    #     mesh_new = o3d.geometry.TriangleMesh()
    #     for idx, name in enumerate(self.urdf_dict[type]["scene"].geometry.keys()):
    #         mesh_new += copy.deepcopy(self.urdf_dict[type]["mesh_list"][idx]).transform(
    #             self.urdf_dict[type]["scene"].graph.get(name)[0]
    #         )
    #     return mesh_new

    # def get_mesh_pointcloud(self, joint_pos):
    #     self.Leap_urdf.update_cfg(joint_pos)
    #     right_mesh = self._update_meshes("right_xhand")  # Get the new updated mesh
    #     robot_pc = right_mesh.sample_points_uniformly(number_of_points=80000)
    #
    #     # Convert the sampled mesh point cloud to the format expected by Open3D
    #     new_points = np.asarray(robot_pc.points)  # Convert to numpy array for points
    #
    #     return new_points
    #
    # def create_target_vis(self):
    #
    #     # load balls (used for visualization)
    #     small_ball_radius = 0.001
    #     small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius) # small ball used to indicate fingertip current position
    #     ball_radius = 0.02
    #     ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius) # large ball used to indicate fingertip goal position
    #     baseMass = 0.001
    #     basePosition = [0, 0, 0]
    #
    #     self.ball1Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition) # for base and finger tip joints
    #     self.ball2Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)
    #     self.ball3Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)
    #     self.ball4Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)
    #     self.ball5Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
    #     self.ball6Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
    #     self.ball7Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
    #     self.ball8Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
    #     self.ball9Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
    #     self.ball10Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
    #
    #     self.ball11Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition) # for base and finger tip joints
    #     self.ball12Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)
    #     self.ball13Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)
    #     self.ball14Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)
    #     self.ball15Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
    #     self.ball16Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
    #     self.ball17Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
    #     self.ball18Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
    #     self.ball19Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
    #     self.ball20Mbt = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
    #
    #     p.changeVisualShape(self.ball1Mbt, -1, rgbaColor=[1, 0, 0, 1])  # Red
    #     p.changeVisualShape(self.ball2Mbt, -1, rgbaColor=[0, 1, 0, 1])  # Green
    #     p.changeVisualShape(self.ball3Mbt, -1, rgbaColor=[0, 0, 1, 1])  # Blue
    #     p.changeVisualShape(self.ball4Mbt, -1, rgbaColor=[1, 1, 0, 1])  # Yellow
    #     p.changeVisualShape(self.ball5Mbt, -1, rgbaColor=[1, 1, 1, 1])  # White
    #     p.changeVisualShape(self.ball6Mbt, -1, rgbaColor=[0, 0, 0, 1])  # Black
    #     p.changeVisualShape(self.ball7Mbt, -1, rgbaColor=[1, 0, 0, 1])  # Red
    #     p.changeVisualShape(self.ball8Mbt, -1, rgbaColor=[0, 1, 0, 1])  # Green
    #     p.changeVisualShape(self.ball9Mbt, -1, rgbaColor=[0, 0, 1, 1])  # Blue
    #     p.changeVisualShape(self.ball10Mbt, -1, rgbaColor=[1, 1, 0, 1])  # Yellow
    #
    #     p.changeVisualShape(self.ball11Mbt, -1, rgbaColor=[1, 0, 0, 1])  # Red
    #     p.changeVisualShape(self.ball12Mbt, -1, rgbaColor=[0, 1, 0, 1])  # Green
    #     p.changeVisualShape(self.ball13Mbt, -1, rgbaColor=[0, 0, 1, 1])  # Blue
    #     p.changeVisualShape(self.ball14Mbt, -1, rgbaColor=[1, 1, 0, 1])  # Yellow
    #     p.changeVisualShape(self.ball15Mbt, -1, rgbaColor=[1, 1, 1, 1])  # White
    #     p.changeVisualShape(self.ball16Mbt, -1, rgbaColor=[0, 0, 0, 1])  # Black
    #     p.changeVisualShape(self.ball17Mbt, -1, rgbaColor=[1, 0, 0, 1])  # Red
    #     p.changeVisualShape(self.ball18Mbt, -1, rgbaColor=[0, 1, 0, 1])  # Green
    #     p.changeVisualShape(self.ball19Mbt, -1, rgbaColor=[0, 0, 1, 1])  # Blue
    #     p.changeVisualShape(self.ball20Mbt, -1, rgbaColor=[1, 1, 0, 1])  # Yellow
    #
    #     no_collision_group = 0
    #     no_collision_mask = 0
    #     p.setCollisionFilterGroupMask(self.ball1Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball2Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball3Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball4Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball5Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball6Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball7Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball8Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball9Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball10Mbt, -1, no_collision_group, no_collision_mask)
    #
    #     p.setCollisionFilterGroupMask(self.ball11Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball12Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball13Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball14Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball15Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball16Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball17Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball18Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball19Mbt, -1, no_collision_group, no_collision_mask)
    #     p.setCollisionFilterGroupMask(self.ball20Mbt, -1, no_collision_group, no_collision_mask)


    # def rest_target_vis(self):
    #     p.resetBaseVelocity(self.ball1Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball2Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball3Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball4Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball5Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball6Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball7Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball8Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball9Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball10Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball11Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball12Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball13Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball14Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball15Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball16Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball17Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball18Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball19Mbt, [0, 0, 0], [0, 0, 0])
    #     p.resetBaseVelocity(self.ball20Mbt, [0, 0, 0], [0, 0, 0])

    def switch_vector_from_xhand(self, vector):
        return [vector[0], -vector[2], vector[1]]

    def post_process_xhand_pos(self, rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos, rightHandPinky_pos):
        rightHandThumb_pos[-1] *= -1.0
        rightHandThumb_pos = self.switch_vector_from_xhand(rightHandThumb_pos)
        rightHandIndex_pos[-1] *= -1.0
        rightHandIndex_pos = self.switch_vector_from_xhand(rightHandIndex_pos)
        rightHandMiddle_pos[-1] *= -1.0
        rightHandMiddle_pos = self.switch_vector_from_xhand(rightHandMiddle_pos)
        rightHandRing_pos[-1] *= -1.0
        rightHandRing_pos = self.switch_vector_from_xhand(rightHandRing_pos)
        rightHandPinky_pos[-1] *= -1.0
        rightHandPinky_pos = self.switch_vector_from_xhand(rightHandPinky_pos)

        return rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos, rightHandPinky_pos

    #クォータニオンのｗｘｙｚの順番を変更
    def post_process_xhand_ori(self, input_quat):
        wxyz_input_quat = np.array([input_quat[3], input_quat[0], input_quat[1], input_quat[2]])
        wxyz_input_mat = quat2mat(wxyz_input_quat)

        rot_mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        wxyz_input_mat = np.dot(wxyz_input_mat, rot_mat)

        wxyz_transform_quat = mat2quat(wxyz_input_mat)
        xyzw_transform_quat = np.array([wxyz_transform_quat[1], wxyz_transform_quat[2], wxyz_transform_quat[3], wxyz_transform_quat[0]])

        return xyzw_transform_quat

    def compute_IK(self, right_hand_pos, right_hand_wrist_ori):
        p.stepSimulation()

        # get right hand position information including fingers
        rightHand_pos = right_hand_pos[0]
        rightHandThumb_pos = (right_hand_pos[4] - rightHand_pos)
        rightHandIndex_pos = (right_hand_pos[8] - rightHand_pos)
        rightHandMiddle_pos = (right_hand_pos[12] - rightHand_pos)
        rightHandRing_pos = (right_hand_pos[16] - rightHand_pos)
        rightHandPinky_pos=(right_hand_pos[20] - rightHand_pos)

        # transform right hand orientation
        rightHand_rot = right_hand_wrist_ori#3×3の回転行列

        #クォータニオンの変換
        # rightHand_rot = self.post_process_xhand_ori(rightHand_rot)
        # euler_angles = quat2euler(np.array([rightHand_rot[3], rightHand_rot[0], rightHand_rot[1], rightHand_rot[2]]))
        # quat_angles = euler2quat(-euler_angles[0], -euler_angles[1], euler_angles[2]).tolist()
        # rightHand_rot = np.array(quat_angles[1:] + quat_angles[:1])
        # rightHand_rot = rotate_quaternion_xyzw(rightHand_rot, np.array([1.0, 0.0, 0.0]), np.pi / 2.0)

        rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos, rightHandPinky_pos = self.post_process_xhand_pos(rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos, rightHandPinky_pos)
        #一旦保留
        # rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos = self.update_target_vis(rightHand_rot, rightHandThumb_pos, rightHandIndex_pos, rightHandMiddle_pos, rightHandRing_pos)

        xhandEndEffectorPos = [
            rightHandIndex_pos,
            rightHandMiddle_pos,
            rightHandRing_pos,
            rightHandThumb_pos,
            rightHandPinky_pos
        ]

        jointPoses = []
        for i in range(5):
            jointPoses = jointPoses + list(
                p.calculateInverseKinematics(self.xhandId, self.xhandEndEffectorIndex[i], xhandEndEffectorPos[i],
                                      lowerLimits=self.hand_lower_limits, upperLimits=self.hand_upper_limits, jointRanges=self.hand_joint_ranges,
                                      restPoses=self.HAND_Q.tolist(), maxNumIterations=1000, residualThreshold=0.001))[4 * i:4 * (i + 1)]
        jointPoses = tuple(jointPoses)



        combined_jointPoses = (jointPoses[0:4] + (0.0,) + jointPoses[4:8] + (0.0,) + jointPoses[8:12] + (0.0,) + jointPoses[12:16] + (0.0,))
        combined_jointPoses = list(combined_jointPoses)

        # update the hand joints
        # for i in range(20):
        #     p.setJointMotorControl2(
        #         bodyIndex=self.LeapId,
        #         jointIndex=i,
        #         controlMode=p.POSITION_CONTROL,
        #         targetPosition=combined_jointPoses[i],
        #         targetVelocity=0,
        #         force=500,
        #         positionGain=0.3,
        #         velocityGain=1,
        #     )
        #
        #
        #
        # p.resetBasePositionAndOrientation(
        #     self.LeapId,
        #     rotate_vector_by_quaternion_using_matrix(self.leap_center_offset, rightHand_rot),
        #     rightHand_rot,
        # )
        #
        #
        #
        # self.rest_target_vis()

        # map results to real robot
        real_right_robot_hand_q = np.array([0.0 for _ in range(11)])
        # real_left_robot_hand_q = np.array([0.0 for _ in range(16)])

        real_right_robot_hand_q[0:3] = jointPoses[0:3]
        real_right_robot_hand_q[3:6] = jointPoses[3:6]
        real_right_robot_hand_q[6:8] = jointPoses[6:8]
        real_right_robot_hand_q[8:10] = jointPoses[8:10]
        real_right_robot_hand_q[10:12] = jointPoses[10:12]


        # real_right_robot_hand_q[0:2] = real_right_robot_hand_q[0:2][::-1]
        # real_right_robot_hand_q[4:6] = real_right_robot_hand_q[4:6][::-1]
        # real_right_robot_hand_q[8:10] = real_right_robot_hand_q[8:10][::-1]



        # generate pointcloud of the left and right hand with forward kinematics
        # right_hand_pointcloud = self.get_mesh_pointcloud(real_right_robot_hand_q)



        return real_right_robot_hand_q


if __name__=="__main__":
    hand=xhandIK()