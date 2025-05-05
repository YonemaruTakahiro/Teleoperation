import threading, csv, os
import time, datetime, getopt, sys
from myclass import myfunction
from myclass.MyDynamixel2 import MyDynamixel
from myclass.MyMagneticSensor import MagneticSensor
import queue
import pickle
from nokov.nokovsdk import *
preFrmNo = 0
curFrmNo = 0
global client



def init_motion_capture():
    serverIp = '10.1.1.198'
    try:
        opts, args = getopt.getopt([],"hs:",["server="])
    except getopt.GetoptError:
        print('NokovSDKClient.py -s <serverIp>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('NokovSDKClient.py -s <serverIp>')
            sys.exit()
        elif opt in ("-s", "--server"):
            serverIp = arg

    print ('serverIp is %s' % serverIp)
    print("Started the Nokov_SDK_Client Demo")
    global client
    client = PySDKClient()

    ver = client.PyNokovVersion()
    
    ret = client.Initialize(bytes(serverIp, encoding = "utf8"))
    if ret == 0:
        print("Connect to the Nokov Succeed")
    else:
        print("Connect Failed: [%d]" % ret)
        exit(0)
        
def get_dynamixel(Motors, motorpath):
    motor_datas = []
    filepath = motorpath
    filenumber = 1
    i = 0
    try:
        while not stop_event.is_set():  # stop_eventがセットされるまでループ
            now_time  = datetime.datetime.now()
            motor_angle = Motors.get_present_angles()
            motor_current = Motors.get_present_currents()
            # formatted_now = now_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            motor_data = myfunction.combine_lists(motor_angle, motor_current)
            motor_data.insert(0, now_time)
            motor_datas.append(motor_data)
            if write_pkl_event_motor.is_set():
                print(motor_data)
                # writing_motor_event.set()
                now = datetime.datetime.now()
                filename = filepath + str(filenumber) + "_" +now.strftime('%Y%m%d_%H%M%S') + '.pickle'
                with open(filename, "wb") as f:
                    pickle.dump(motor_datas, f)  # motor_datasをpickleで保存
                write_pkl_event_motor.clear()  # write_pkl_eventをリセット
                filenumber = filenumber + 1
                # writing_motor_event.clear()
    finally:
        thread_name = threading.current_thread().name
        results[thread_name] = motor_datas

# daemon=Trueで強制終了

def get_magsensor(Ms, magpath):
    Mag_datas = []
    filepath = magpath
    filenumber = 1
    i = 0
    try:
        while not stop_event.is_set():  # stop_eventがセットされるまでループ\
            now_time  = datetime.datetime.now()
            mag_data = Ms.get_value()
            # formatted_now = now_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            mag_data = [mag_data]
            mag_data.insert(0, now_time)
            Mag_datas.append(mag_data)
            if write_pkl_event_mag.is_set():
                print(mag_data)
                # writing_mag_event.set()
                now = datetime.datetime.now()
                filename = filepath + str(filenumber) + "_"+ now.strftime('%Y%m%d_%H%M%S') + '.pickle'
                with open(filename, "wb") as f:
                    pickle.dump(Mag_datas, f)  # motor_datasをpickleで保存
                write_pkl_event_mag.clear()  # write_pkl_eventをリセット
                filenumber = filenumber + 1
                # writing_mag_event.clear()
    finally:
        thread_name = threading.current_thread().name
        results[thread_name] = Mag_datas

    

def move(Motors, howtomovepath):
    with open(howtomovepath, mode='br') as fi:
        change_angle = pickle.load(fi)
    len_angle = len(change_angle)
    print(change_angle)
    for i, angles in enumerate(change_angle):
        print(angles)
        print(str(i) + "/" +  str(len_angle))
        Motors.move_to_points(angles, times = 7)
        # Motors.move_to_point(1, angles[0])
        # Motors.move_to_point(2, angles[1])
        # Motors.move_to_point(3, angles[2])
        # Motors.move_to_point(4, angles[3])
        if (i +1) % 500 == 0:
            write_pkl_event_motor.set()
            write_pkl_event_mag.set()
            write_pkl_event_Mc.set()
            time.sleep(5)

    # time.sleep(2)
    stop_event.set()
    
    
    
def py_data_func(pFrameOfMocapData, pUserData):
    if pFrameOfMocapData == None:  
        print("Not get the data frame.\n")
    else:
        frameData = pFrameOfMocapData.contents
        global preFrmNo, curFrmNo 
        curFrmNo = frameData.iFrame
        if curFrmNo == preFrmNo:
            return
        global client
        preFrmNo = curFrmNo

        length = 128
        szTimeCode = bytes(length)
        client.PyTimecodeStringify(frameData.Timecode, frameData.TimecodeSubframe, szTimeCode, length)
        motiondata = [datetime.datetime.now()]
        for iMarkerSet in range(frameData.nMarkerSets):
            markerset = frameData.MocapData[iMarkerSet]
            for iMarker in range(markerset.nMarkers):
                motiondata.extend([markerset.Markers[iMarker][0],markerset.Markers[iMarker][1], markerset.Markers[iMarker][2]] )
    return motiondata
            


def get_motioncapture(Ms, mcpath):
    Motion_datas = []
    filepath = mcpath
    filenumber = 1
    i = 0
    try:
        while not stop_event.is_set():  # stop_eventがセットされるまでループ\
            frame = client.PyGetLastFrameOfMocapData()
            if frame :
                try:
                    motiondata = py_data_func(frame, client)
                    Motion_datas.append(motiondata)
                finally:
                    client.PyNokovFreeFrame(frame)
            if write_pkl_event_Mc.is_set():
                print(motiondata)
                # writing_mag_event.set()
                now = datetime.datetime.now()
                filename = filepath + str(filenumber) + "_"+ now.strftime('%Y%m%d_%H%M%S') + '.pickle'
                with open(filename, "wb") as f:
                    pickle.dump(Motion_datas, f)  # motor_datasをpickleで保存
                write_pkl_event_Mc.clear()  # write_pkl_eventをリセット
                filenumber = filenumber + 1
                # writing_mag_event.clear()
    finally:
        thread_name = threading.current_thread().name
        results[thread_name] = Motion_datas
        
        



# ----------------------------------------------------------------------------------------


result_dir = r"temp"

base_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult"

# ----------------------------------------------------------------------------------------







base_path = os.path.join(base_path, result_dir)
print(base_path)
motorpath = os.path.join(base_path, "motor_")
magpath = os.path.join(base_path, "mag_")
mcpath = os.path.join(base_path, "mc_")
howtomovepath = myfunction.find_pickle_files("howtomove", base_path)

init_motion_capture()
Motors = MyDynamixel()
Ms = MagneticSensor()
results = {}
stop_event = threading.Event()
write_pkl_event_motor = threading.Event()
write_pkl_event_mag = threading.Event()
write_pkl_event_Mc = threading.Event()

print("◆スレッド:",threading.current_thread().name)

thread1 = threading.Thread(target=get_dynamixel, args=(Motors,motorpath,), name="motor")
thread2 = threading.Thread(target=get_magsensor, args=(Ms,magpath,), name="magsensor")
thread3 = threading.Thread(target=move, args=(Motors,howtomovepath,), name="move")
thread4 = threading.Thread(target=get_motioncapture, args=(Ms,mcpath,), name="motioncapture")

thread1.start()
thread2.start()
thread3.start()
thread4.start()

thread1.join()
thread2.join()
thread3.join()
thread4.join()


for key, value in results.items():
    filename = key
    now = datetime.datetime.now()
    filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.pickle'
    with open(filename, 'wb') as fo:
        pickle.dump(value, fo)
    





print("Results:", results)