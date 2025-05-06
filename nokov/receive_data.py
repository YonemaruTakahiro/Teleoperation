import time, datetime, getopt, sys
from nokov.nokovsdk import *


preFrmNo = 0
curFrmNo = 0

def py_msg_func(iLogLevel, szLogMessage):
    szLevel = "None"
    if iLogLevel == 4:
        szLevel = "Debug"
    elif iLogLevel == 3:
        szLevel = "Info"
    elif iLogLevel == 2:
        szLevel = "Warning"
    elif iLogLevel == 1:
        szLevel = "Error"

    print("[%s] %s" % (szLevel, cast(szLogMessage, c_char_p).value))


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
        print(f"\nFrameNo: {frameData.iFrame}\tTimeStamp:{frameData.iTimeStamp}\t Timecode:{szTimeCode.decode('utf-8')}")
        print(f"MarkerSet [Count={frameData.nMarkerSets}]")
        for iMarkerSet in range(frameData.nMarkerSets):
            markerset = frameData.MocapData[iMarkerSet]
            print(
                f"Markerset{iMarkerSet + 1}: {markerset.szName.decode('utf-8')} [nMarkers Count={markerset.nMarkers}]")
            print("{")

            for iMarker in range(markerset.nMarkers):
                print(f"\tMarker{iMarker + 1}(mm) \tx:{markerset.Markers[iMarker][0]:6.2f}" \
                      f"\ty:{markerset.Markers[iMarker][1]:6.2f}\tz:{markerset.Markers[iMarker][2]:6.2f}")
            print("}")

        print(f"Markerset.RigidBodies [Count={frameData.nRigidBodies}]")
        #これを使う必要がある
        for iBody in range(frameData.nRigidBodies):
            body = frameData.RigidBodies[iBody]
            print("{")
            print(f"\tid:{body.ID}")
            print(f"\t    (mm)\tx:{body.x:6.2f}\ty:{body.y:6.2f}\tz:{body.z:6.2f}")
            print(f"\t\t\tqx:{body.qx:6.2f}\tqy:{body.qy:6.2f}\tqz:{body.qz:6.2f}\tqw:{body.qw:6.2f}")

            # Markers
            print(f"\tRigidBody markers [Count={body.nMarkers}]")
            for i in range(body.nMarkers):
                marker = body.Markers[i]
                print(f"\tMarker{body.MarkerIDs[i]}(mm)\tx:{marker[0]:6.2f}\ty:{marker[1]:6.2f}\tz:{marker[2]:6.2f}")
            print("}")

        return frameData.RigidBodies

        # print(f"Markerset.Skeletons [Count={frameData.nSkeletons}]")
        # #これは使わない
        # for iSkeleton in range(frameData.nSkeletons):
        #     # Segments
        #     skeleton = frameData.Skeletons[iSkeleton]
        #     print(f"nSegments Count={skeleton.nRigidBodies}")
        #     print("{")
        #     for iBody in range(skeleton.nRigidBodies):
        #         body = skeleton.RigidBodyData[iBody]
        #         print(f"\tSegment id:{body.ID}")
        #         print(f"\t    (mm)\tx:{body.x:6.2f}\ty:{body.y:6.2f}\tz:{body.z:6.2f}")
        #         print(f"\t\t\tqx:{body.qx:6.2f}\tqy:{body.qy:6.2f}\tqz:{body.qz:6.2f}\tqz:{body.qw:6.2f}")
        #         for iMarkerIndex in range(body.nMarkers):
        #             marker = body.Markers[iMarkerIndex]
        #             print(f"\tMarker{body.MarkerIDs[iMarkerIndex]}(mm)" f"\tx:{marker[0]:6.2f}\ty:{marker[1]:6.2f}\tz:{marker[2]:6.2f}")
        #     print("}\n")
        #
        # # Unidentified Markers
        # #これは使わない
        # print(f"nUnidentified Markers [Count={frameData.nOtherMarkers}]")
        # print("{")
        # for i in range(frameData.nOtherMarkers):
        #     otherMarker = frameData.OtherMarkers[i]
        #     print(f"\tUnidentified Markers{i + 1}(mm)"
        #           f"\tX:{otherMarker[0]:6.2f}\tY:{otherMarker[1]:6.2f}\tZ:{otherMarker[2]:6.2f}")
        #
        # print("}\n")
        #
        # # Analog values
        # #これは使わない
        # print(f"nAnalog [Count={frameData.nAnalogdatas}]")
        # print("{")
        # for i in range(frameData.nAnalogdatas):
        #     print(f"\tAnalogData {i + 1}: {frameData.Analogdata[i]:6.3f}")
        # print("}\n")



if __name__=="__main__":
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
        t1=time.time()
        frame = client.PyGetLastFrameOfMocapData()
        if frame:

            try:
                t1 = time.time()
                py_data_func(frame, client)
                t2 = time.time()
                print(f"周期:{t2 - t1}")

            finally:
                client.PyNokovFreeFrame(frame)


