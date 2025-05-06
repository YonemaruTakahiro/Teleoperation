import multiprocessing
import time
import numpy as np
from queue import Empty
from wrs import wd, rm, ur3d, rrtc, mgm, mcm
from wrs.robot_sim.end_effectors.multifinger.xhand import xhand_right as xhr
from wrs.robot_con.xhand import xhand_x as xhx
from nokov.receive_data import *
from utils.precise_sleep import precise_wait

WIDTH = 1280
HEIGHT = 720


class nokov_data:
    def __init__(self, x=None, y=None, z=None, qx=None, qy=None, qz=None, qw=None):
        self.x = x
        self.y = y
        self.z = z
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw


def receive_data(queue1: multiprocessing.Queue):
    serverIp = '10.1.1.198'

    try:
        opts, args = getopt.getopt([], "hs:", ["server="])
    except getopt.GetoptError:
        print('NokovrSDKClient.py -s <serverIp>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('NokovrSDKClient.py -s <serverIp>')
            sys.exit()
        elif opt in ("-s", "--server"):
            serverIp = arg

    print('serverIp is %s' % serverIp)
    print("Started the Nokovr_SDK_Client Demo")
    client = PySDKClient()

    ver = client.PyNokovVersion()
    print('NokovrSDK Sample Client 2.4.0.5428(NokovrSDK ver. %d.%d.%d.%d)' % (ver[0], ver[1], ver[2], ver[3]))

    client.PySetVerbosityLevel(0)
    client.PySetMessageCallback(py_msg_func)
    client.PySetDataCallback(py_data_func, None)

    print("Begin to init the SDK Client")
    ret = client.Initialize(bytes(serverIp, encoding="utf8"))

    if ret == 0:
        print("Connect to the Nokov Succeed")
    else:
        print("Connect Failed: [%d]" % ret)
        exit(0)
    while True:
        t1 = time.time()
        frame = client.PyGetLastFrameOfMocapData()
        if frame:

            try:
                t1 = time.time()
                q = py_data_func(f0rame, client)
                queue1.put(q)
                t2 = time.time()
                print(f"周期:{t2 - t1}")

            finally:
                client.PyNokovFreeFrame(frame)


def xarm(queue1: multiprocessing.Queue):
    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)

    nokov_data_base = nokov_data()
    nokov_data_hand = nokov_data()

    def update(nokov_data_base, nokov_data_hand, task):

        q = queue1.get(timeout=20)

        if q is not None:
            nokov_data_base.x=q[0].x*1000
            nokov_data_base.y=q[0].y*1000
            nokov_data_base.z=q[0].z*1000
            nokov_data_base.qx=q[0].qx*1000
            nokov_data_base.qy=q[0].qy*1000
            nokov_data_base.qz=q[0].qz*1000
            nokov_data_base.qw=q[0].qw*1000

            nokov_data_hand.x = q[1].x * 1000
            nokov_data_hand.y = q[1].y * 1000
            nokov_data_hand.z = q[1].z * 1000
            nokov_data_hand.qx = q[1].qx * 1000
            nokov_data_hand.qy = q[1].qy * 1000
            nokov_data_hand.qz = q[1].qz * 1000
            nokov_data_hand.qw = q[1].qw * 1000


        return task.again

    taskMgr.doMethodLater(0.0, update, "update",
                          extraArgs=[nokov_data_base, nokov_data_hand],
                          appendTask=True)

    base.run()


if __name__ == "__main__":
    queue1 = multiprocessing.Queue(maxsize=1)

    process1 = multiprocessing.Process(target=receive_data, args=(queue1,))
    process2 = multiprocessing.Process(target=xarm, args=(queue1,))

    process1.start()
    process2.start()

    process1.join()
    process2.join()

    print("done")
