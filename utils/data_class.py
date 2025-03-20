import numpy as np

class animation:
    def __init__(self, tgt_pos, tgt_rotmat, jnt_values):
        self.tgt_pos = tgt_pos
        self.tgt_rotmat = tgt_rotmat
        self.jaw_width = 0
        self.current_jnt_values = jnt_values
        self.next_jnt_values = None
        # self.next_next_jnt_values = None
        self.current_jaw_width=0
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

class animation_sim:
    def __init__(self,tgt_pos,tgt_rotmat,jnt_values,mesh_model,robot_model):
        self.tgt_pos=tgt_pos
        self.tgt_rotmat=tgt_rotmat
        self.jaw_width = 0
        self.current_jnt_values = jnt_values
        self.next_jnt_values = None
        self.current_jaw_width = 0
        self.mesh_model=mesh_model
        self.robot_model = robot_model
        self.count=0

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

class handdata:
    def __init__(self, hand_pos=None, hand_rotmat=None, jaw_width=None, detected_time=None):
        self.hand_pos = hand_pos
        self.hand_rotmat = hand_rotmat
        self.jaw_width = jaw_width
        self.detected_time = detected_time


class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data