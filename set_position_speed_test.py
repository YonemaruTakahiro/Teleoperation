import xarm_realtime_con as xrc
import numpy as np

from wrs.robot_sim.robots.xarm7_dual import xarm7_dual as x7xh


if __name__ == "__main__":
    robot = x7xh.XArm7Dual(enable_cc=True)
    robotx = xrc.xhand_XArmX(ip="192.168.1.205")

    start_manipulator_conf = robotx.arm_get_jnt_values()
    # start_manipulator_pos, start_manipulator_rotmat = robot.fk(np.array(start_manipulator_conf), toggle_jacobian=False)
    start_manipulator_pos=np.array([ 0.50463533, -0.55799959,  0.20599448])
    start_manipulator_rotmat=np.array([[-0.97211256, 0.06344848,  0.22576858],[-0.05651785, -0.99771439, 0.03703683],[ 0.2276025,   0.02324402,  0.97347667]])
    # print(f"start_manipulator_pos,start_manipulator_rotmat:{start_manipulator_pos},{start_manipulator_rotmat}")

    robotx.set_position_wrs([start_manipulator_pos, start_manipulator_rotmat],
                            end_effector_center_pos=robot._rgt_arm.manipulator.loc_tcp_pos,
                            end_effector_ceter_rotmat=robot._rgt_arm.manipulator.loc_tcp_rotmat, speed=500, wait=True)

    tgt_pos=start_manipulator_pos
    tgt_pos[1]+=0.4

    tgt_rotmat=start_manipulator_rotmat

    robotx.set_position_wrs([tgt_pos, tgt_rotmat],
                            end_effector_center_pos=robot._rgt_arm.manipulator.loc_tcp_pos,
                            end_effector_ceter_rotmat=robot._rgt_arm.manipulator.loc_tcp_rotmat, speed=500, wait=True)