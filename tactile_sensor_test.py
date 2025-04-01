from hand_detector_multi_finger import HandDetector_multifinger
from xhand_class_ikpy import xhand_K
from utils.data_class import animation, WiLor_Data, Data
from utils.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

import pickle
from pathlib import Path
import pyrealsense2 as rs
import multiprocessing
import torch
import time
import cv2
import numpy as np
from queue import Empty
from loguru import logger
from wrs import wd, rm, ur3d, rrtc, mgm, mcm
from wrs.robot_sim.end_effectors.multifinger.xhand import xhand_right as xhr
from wrs.robot_con.xhand import xhand_x as xhx

from ultralytics import YOLO
from utils.precise_sleep import precise_wait
from wilor.models import WiLoR, load_wilor
from wilor.utils.renderer import Renderer, cam_crop_to_full



xhexe = xhx.XHandX("/dev/ttyUSB0")
    xhand_k = xhand_K()