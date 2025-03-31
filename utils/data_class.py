import numpy as np


class animation:
    def __init__(self, tgt_pos, tgt_rotmat, jnt_values):
        self.tgt_pos = tgt_pos
        self.tgt_rotmat = tgt_rotmat
        self.jaw_width = 0
        self.current_jnt_values = jnt_values
        self.next_jnt_values = None
        # self.next_next_jnt_values = None
        self.current_jaw_width = 0
        # self.jnts_velocity = np.zeros(6)  ##６軸に設定

    @staticmethod
    def rotmat_error(np_verts1, np_vert2):
        e = np_verts1 - np_vert2
        e = np.square(e)
        e = np.sum(e)
        return e / e.size

    @staticmethod
    def pos_error(np_verts1, np_vert2):
        e = np_verts1 - np_vert2
        e = np.square(e)
        e = np.sum(e)
        return e


class multi_finger_animation:
    def __init__(self, tgt_pos, tgt_rotmat, manipulator_jnt_values, xhand_jnt_values, mesh_model, robot_model):
        self.tgt_pos = tgt_pos
        self.tgt_rotmat = tgt_rotmat
        self.thumb_position = None
        self.index_position = None
        self.middle_position = None
        self.ring_position = None
        self.pinky_position = None
        self.current_manipulator_jnt_values = manipulator_jnt_values
        self.next_manipulator_jnt_values = None
        self.current_xhand_jnt_values = xhand_jnt_values
        self.next_xhand_jnt_values = None
        self.current_jaw_width = 0
        self.mesh_model = mesh_model
        self.robot_model = robot_model
        self.count = 0

    @staticmethod
    def rotmat_error(np_verts1, np_vert2):
        e = np_verts1 - np_vert2
        e = np.square(e)
        e = np.sum(e)
        return e / e.size

    @staticmethod
    def pos_error(np_verts1, np_vert2):
        e = np_verts1 - np_vert2
        e = np.square(e)
        e = np.sum(e)
        return e


class animation_sim:
    def __init__(self, tgt_pos, tgt_rotmat, jnt_values, mesh_model, robot_model):
        self.tgt_pos = tgt_pos
        self.tgt_rotmat = tgt_rotmat
        self.jaw_width = 0
        self.current_jnt_values = jnt_values
        self.next_jnt_values = None
        self.current_jaw_width = 0
        self.mesh_model = mesh_model
        self.robot_model = robot_model
        self.count = 0

    @staticmethod
    def rotmat_error(np_verts1, np_vert2):
        e = np_verts1 - np_vert2
        e = np.square(e)
        e = np.sum(e)
        return e / e.size

    @staticmethod
    def pos_error(np_verts1, np_vert2):
        e = np_verts1 - np_vert2
        e = np.square(e)
        e = np.sum(e)
        return e


class WiLor_Data:
    def __init__(self, eef_pos=None, eef_rotmat=None, keypoints_3d=None,human_hand_rotmat=None):
        self.eef_pos = eef_pos
        self.eef_rotmat = eef_rotmat
        self.keypoints_3d = keypoints_3d
        self.human_hand_rotmat=human_hand_rotmat



class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data
