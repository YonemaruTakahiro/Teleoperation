import numpy as np
from wrs import wd, rm, ur3d, rrtc, mgm, mcm
from wrs.robot_sim.robots.xarm7_dual import xarm7_dual as x7xh
import xarm_realtime_con as xrc

if __name__ == "__main__":
    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)

    robot = x7xh.XArm7Dual(enable_cc=True)
    robotx = xrc.xhand_XArmX(ip="192.168.1.205")
    start_manipulator_conf = robotx.arm_get_jnt_values()
    # print(f"start_manipulator:{start_manipulator_conf}")
    start_manipulator_pos, start_manipulator_rotmat = robot.fk(np.array(start_manipulator_conf), toggle_jacobian=False)

    tgt_euler = rm.rotmat_to_euler(start_manipulator_rotmat, "sxyz")

    pose = robotx.get_position_wrs(end_effector_center_pos=robot._rgt_arm.manipulator.loc_tcp_pos,
                                   end_effector_ceter_rotmat=robot._rgt_arm.manipulator.loc_tcp_rotmat)

    start_robot_model = robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True, toggle_jnt_frames=False)
    start_mesh_model = mgm.gen_frame(pos=start_manipulator_pos, rotmat=start_manipulator_rotmat)
    # mgm.gen_frame(pos=position, rotmat=rot).attach_to(base)

    # pose[0]+=40
    robotx.set_position_wrs(pose,end_effector_center_pos=robot._rgt_arm.manipulator.loc_tcp_pos,end_effector_ceter_rotmat=robot._rgt_arm.manipulator.loc_tcp_rotmat)
    # robotx.set_position(pose,speed=80,is_radian=True)#ラジアンに設定する必要あり
    start_mesh_model.attach_to(base)
    start_robot_model.attach_to(base)

    base.run()
