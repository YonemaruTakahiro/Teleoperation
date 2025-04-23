import math
from wrs import wd, rm, ur3d, rrtc, mgm, mcm
import time
import numpy as np
from wrs.drivers.xarm.wrapper import xarm_api as arm
import wrs.motion.trajectory.totg as pwp


class xhand_XArmX(object):

    def __init__(self, ip="10.2.0.203"):
        """
        :param _arm_x: an instancde of arm.XArmAPI
        :return:
        """
        super().__init__()
        self._arm_x = arm.XArmAPI(port=ip)
        if self._arm_x.has_err_warn:
            if self._arm_x.get_err_warn_code()[1][0] == 1:
                print("The Emergency Button is pushed in to stop!")
                input("Release the emergency button and press any key to continue. Press Enter to continue...")
        self._arm_x.clean_error()
        self._arm_x.clean_error()
        self._arm_x.motion_enable()
        self._arm_x.set_mode(0)  # servo motion mode
        self._arm_x.set_state(state=0)
        # self._arm_x.set_tcp_max_velocity(200, 500)

    @property
    def arm(self):
        return self._arm_x

    def get_position_wrs(self, end_effector_center_pos, end_effector_ceter_rotmat):
        code, pose = self._arm_x.get_position(is_radian=True)
        if code != 0:
            raise Exception(f"The returned code of get_position is wrong! Code: {code}")
        gl_flange_rot = np.array(rm.rotmat_from_euler(pose[3], pose[4], pose[5], order="sxyz"))
        pos = np.array(pose[:3]) / 1000 + gl_flange_rot @ end_effector_center_pos
        rot = gl_flange_rot @ end_effector_ceter_rotmat

        return pos, rot

    def arm_get_jnt_values(self):
        code, jnt_values = self._arm_x.get_servo_angle(is_radian=True)
        if code != 0:
            raise Exception(f"The returned code of get_servo_angle is wrong! Code: {code}")
        return np.asarray(jnt_values)

    def set_position_wrs(self, wrs_pose, end_effector_center_pos, end_effector_ceter_rotmat, speed=None, mvtime=None,
                         wait=True):
        if len(wrs_pose) != 2:
            raise Exception(f"pose list format is wrong !!")

        xarm_flange_rotmat = wrs_pose[1] @ end_effector_ceter_rotmat.T
        xarm_flange_pos = wrs_pose[0] - xarm_flange_rotmat @ end_effector_center_pos
        xarm_flange_pos = xarm_flange_pos * 1000

        euler = rm.rotmat_to_euler(xarm_flange_rotmat, order="sxyz")

        self._arm_x.set_position(x=xarm_flange_pos[0], y=xarm_flange_pos[1], z=xarm_flange_pos[2], roll=euler[0],
                                 pitch=euler[1], yaw=euler[2], speed=speed, mvtime=mvtime,
                                 is_radian=True, wait=wait)

    def set_servo_cartesian_wrs(self, wrs_pose, end_effector_center_pos, end_effector_ceter_rotmat, speed=None,
                                mvtime=None, wait=True):
        if len(wrs_pose) != 2:
            raise Exception(f"pose list format is wrong !!")

        xarm_flange_rotmat = wrs_pose[1] @ end_effector_ceter_rotmat.T
        xarm_flange_pos = wrs_pose[0] - xarm_flange_rotmat @ end_effector_center_pos
        xarm_flange_pos = xarm_flange_pos * 1000

        euler = rm.rotmat_to_euler(xarm_flange_rotmat, order="sxyz")

        tgt_pose = [xarm_flange_pos[0], xarm_flange_pos[1], xarm_flange_pos[2], euler[0], euler[1], euler[2]]

        self._arm_x.set_servo_cartesian(mvpose=tgt_pose, speed=speed, mvtime=mvtime,
                                        is_radian=True, wait=wait)

    # def arm_move_jspace_path(self,
    #                          path,
    #                          max_jntvel=None,
    #                          max_jntacc=None,
    #                          start_frame_id=1,
    #                          ctrl_freq=.005):
    #     """
    #     :param path: [jnt_values0, jnt_values1, ...], results of motion planning
    #     :param max_jntvel:
    #     :param max_jntacc:
    #     :param start_frame_id:
    #     :return:
    #     """
    #     if not path or path is None:
    #         raise ValueError("The given is incorrect!")
    #     interp_time, interp_confs, interp_spds, interp_accs = pwp.time_optimal_trajectory_generation(path=path,
    #                                                                                                  max_vels=max_jntvel,
    #                                                                                                  max_accs=max_jntacc,
    #                                                                                                  ctrl_freq=ctrl_freq)
    #     print(f"interp_confs:{interp_confs}")
    #     # interpolated_path = interp_confs[start_frame_id:]
    #     # print(f"interpolated_path:{interpolated_path}")
    #     for jnt_values in interp_confs:
    #         self._arm_x.set_servo_angle_j(jnt_values, mvtime=0.1,is_radian=True)
    #     return

# if __name__ == "__main__":
#     import keyboard
#     from wrs import basis as rm, drivers as arm, robot_sim as rbt
#     import wrs.visualization.panda.world as wd
#
#     base = wd.World(cam_pos=[3, 1, 1.5], lookat_pos=[0, 0, 0.7])
#     rbt_s = rbt.XArmShuidi()
#     rbt_x = XArmShuidiX(ip="10.2.0.203")
#     jnt_values = rbt_x.arm_get_jnt_values()
#     print(jnt_values)
#     jawwidth = rbt_x.arm_get_jaw_width()
#     rbt_s.fk(jnt_values=jnt_values)
#     rbt_s.jaw_to(jawwidth=jawwidth)
#     rbt_s.gen_meshmodel().attach_to(base)
#     # base.run()
#     # rbt_x.agv_move(agv_linear_speed=-.1, agv_angular_speed=.1, time_intervals=5)
#     agv_linear_speed = .2
#     agv_angular_speed = .5
#     arm_linear_speed = .02
#     arm_angular_speed = .05
#     while True:
#         pressed_keys = {"w": keyboard.is_pressed('w'),
#                         "a": keyboard.is_pressed('a'),
#                         "s": keyboard.is_pressed('s'),
#                         "d": keyboard.is_pressed('d'),
#                         "r": keyboard.is_pressed('r'),  # x+ global
#                         "t": keyboard.is_pressed('t'),  # x- global
#                         "f": keyboard.is_pressed('f'),  # y+ global
#                         "g": keyboard.is_pressed('g'),  # y- global
#                         "v": keyboard.is_pressed('v'),  # z+ global
#                         "b": keyboard.is_pressed('b'),  # z- global
#                         "y": keyboard.is_pressed('y'),  # r+ global
#                         "u": keyboard.is_pressed('u'),  # r- global
#                         "h": keyboard.is_pressed('h'),  # p+ global
#                         "j": keyboard.is_pressed('j'),  # p- global
#                         "n": keyboard.is_pressed('n'),  # yaw+ global
#                         "m": keyboard.is_pressed('m')}  # yaw- global
#         # "R": keyboard.is_pressed('R'),  # x+ local
#         # "T": keyboard.is_pressed('T'),  # x- local
#         # "F": keyboard.is_pressed('F'),  # y+ local
#         # "G": keyboard.is_pressed('G'),  # y- local
#         # "V": keyboard.is_pressed('V'),  # z+ local
#         # "B": keyboard.is_pressed('B'),  # z- local
#         # "Y": keyboard.is_pressed('Y'),  # r+ local
#         # "U": keyboard.is_pressed('U'),  # r- local
#         # "H": keyboard.is_pressed('H'),  # p+ local
#         # "J": keyboard.is_pressed('J'),  # p- local
#         # "N": keyboard.is_pressed('N'),  # yaw+ local
#         # "M": keyboard.is_pressed('M')}  # yaw- local
#         values_list = list(pressed_keys.values())
#         if pressed_keys["w"] and pressed_keys["a"]:
#             rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
#         elif pressed_keys["w"] and pressed_keys["d"]:
#             rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
#         elif pressed_keys["s"] and pressed_keys["a"]:
#             rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
#         elif pressed_keys["s"] and pressed_keys["d"]:
#             rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
#         elif pressed_keys["w"] and sum(values_list) == 1:  # if key 'q' is pressed
#             rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=.0, time_interval=.5)
#         elif pressed_keys["s"] and sum(values_list) == 1:  # if key 'q' is pressed
#             rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=.0, time_interval=.5)
#         elif pressed_keys["a"] and sum(values_list) == 1:  # if key 'q' is pressed
#             rbt_x.agv_move(linear_speed=.0, angular_speed=agv_angular_speed, time_interval=.5)
#         elif pressed_keys["d"] and sum(values_list) == 1:  # if key 'q' is pressed
#             rbt_x.agv_move(linear_speed=.0, angular_speed=-agv_angular_speed, time_interval=.5)
#         elif any(pressed_keys[item] for item in ['r', 't', 'f', 'g', 'v', 'b', 'y', 'u', 'h', 'j', 'n', 'm']) and \
#                 sum(values_list) == 1:  # global
#             tic = time.time()
#             current_arm_tcp_pos, current_arm_tcp_rotmat = rbt_s.get_gl_tcp()
#             rel_pos = np.zeros(3)
#             rel_rotmat = np.eye(3)
#             if pressed_keys['r']:
#                 rel_pos = np.array([arm_linear_speed * .5, 0, 0])
#             elif pressed_keys['t']:
#                 rel_pos = np.array([-arm_linear_speed * .5, 0, 0])
#             elif pressed_keys['f']:
#                 rel_pos = np.array([0, arm_linear_speed * .5, 0])
#             elif pressed_keys['g']:
#                 rel_pos = np.array([0, -arm_linear_speed * .5, 0])
#             elif pressed_keys['v']:
#                 rel_pos = np.array([0, 0, arm_linear_speed * .5])
#             elif pressed_keys['b']:
#                 rel_pos = np.array([0, 0, -arm_linear_speed * .5])
#             elif pressed_keys['y']:
#                 rel_rotmat = rm.rotmat_from_euler(arm_angular_speed * .5, 0, 0)
#             elif pressed_keys['u']:
#                 rel_rotmat = rm.rotmat_from_euler(-arm_angular_speed * .5, 0, 0)
#             elif pressed_keys['h']:
#                 rel_rotmat = rm.rotmat_from_euler(0, arm_angular_speed * .5, 0)
#             elif pressed_keys['j']:
#                 rel_rotmat = rm.rotmat_from_euler(0, -arm_angular_speed * .5, 0)
#             elif pressed_keys['n']:
#                 rel_rotmat = rm.rotmat_from_euler(0, 0, arm_angular_speed * .5)
#             elif pressed_keys['m']:
#                 rel_rotmat = rm.rotmat_from_euler(0, 0, -arm_angular_speed * .5)
#             new_arm_tcp_pos = current_arm_tcp_pos + rel_pos
#             new_arm_tcp_rotmat = rel_rotmat.dot(current_arm_tcp_rotmat)
#             last_jnt_values = rbt_s.get_jnt_values()
#             new_jnt_values = rbt_s.ik(tgt_pos=new_arm_tcp_pos,
#                                       tgt_rotmat=new_arm_tcp_rotmat,
#                                       seed_jnt_values=last_jnt_values)
#             if new_jnt_values is not None:
#                 print(new_jnt_values)
#                 print(last_jnt_values)
#                 max_change = np.max(new_jnt_values-last_jnt_values)
#                 print(max_change)
#                 # rbt_s.fk(jnt_values=new_jnt_values)
#                 # rbt_s.jaw_to(ee_values=ee_values)
#                 # rbt_s.gen_meshmodel().attach_to(base)
#                 # base.run()
#             else:
#                 continue
#             rbt_s.fk(jnt_values=new_jnt_values)
#             toc = time.time()
#             start_frame_id = math.ceil((toc - tic) / .01)
#             rbt_x.arm_move_jspace_path([last_jnt_values, new_jnt_values],
#                                        start_frame_id=start_frame_id)
