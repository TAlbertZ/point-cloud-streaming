import numpy as np
from enum import Enum

USING_RUMA = False
RUMA_SCALABLE_CODING = False

MMSYS_HYBRID_TILING = False

SAVE_WHEN_PLOTTING = True

SHOW_ANGULAR_RESOL = True

WEIGHTED_AVERAGE_ANG_RESOL = True


class ActionInitType(Enum):
    LOW = 0
    HIGH = 1
    RANDOM_UNIFORM = 2


class Algo(Enum):
    KKT = 0
    RUMA_NONSCALABLE = 1
    RUMA_SCALABLE = 2
    MMSYS_HYBRID_TILING = 3
    ILQR = 4


class FrameWeightType(Enum):
    CONST = 0  # frame weight = 1 for any frame
    LINEAR_DECREASE = 1
    EXP_DECREASE = 2
    FOV_PRED_ACCURACY = 3  # based on fov overlap ratio between pred and true
    STEP_FUNC = 4
    BELL_SHAPE = 5

SCALABLE_CODING = True

RADIAN_TO_DEGREE = 180 / np.pi

WEIGHT_VARIANCE = 0

NUM_LQR_ITERATIONS = 10

TILE_UTILITY_COEF = np.array(
    [-0.7834842486065501, -0.03049975, 2.78785111, -1.4918287,
     0.26403939])  # a0 is intercept, a0~a4

FRAME_WEIGHT_TYPE = FrameWeightType.EXP_DECREASE
MAX_FRAME_WEIGHT = 1e9
MIN_FRAME_WEIGHT = 1

FRAME_WEIGHT_STEP_IDX = 60

FRAME_WEIGHT_EXP_BOTTOM = 10
FRAME_WEIGHT_EXP_FACTOR = 6

FRAME_WEIGHT_PEAK_IDX = 60


ALGO = Algo.KKT
ACTION_INIT_TYPE = ActionInitType.LOW

SVC_OVERHEAD = 1

SMOOTH_MIN_PARAM = 50

FPS = 30

# FIXME currently if latency larger than 300 will have problem
TARGET_LATENCY = 600  # in frame
ILQR_HORIZON = TARGET_LATENCY // FPS

# Assume frame is independently encoded/decoded
# actual buffer length should be TARGET_LATENCY - 1 frames
BUFFER_LENGTH = TARGET_LATENCY  # in frame

# cannot update for some contents in front of buffer,
# because cannot they finish updating before being played
TIME_GAP_BETWEEN_NOW_AND_FIRST_FRAME_TO_UPDATE = 1  # in second

UPDATE_FREQ = 1  # in second

# according to average values in file '../psnr_weights/byte.pkl'
BYTE_SIZES = [802, 2441, 7158]
# BYTE_SIZES = [127, 467, 1532] # originally, according to the img sent from Yixiang

# assume max tile size is the same across all tiles,
# later should be modified
MAX_TILE_SIZE = BYTE_SIZES[2]  # bytes

# quality parameters a and b
# generated randomly accoding to normal distribution
QUALITY_PARA = []

# each frame in buffer has a weight for update,
# here assume ther're fixed, later should be modified as:
# adapt to dynamic conditions (bandwidth and fov evolution)
FRAME_WEIGHT_IN_BUFFER = []

# 1 / (1 + exp(-c * distance)),
# distance is between viewpoint and each tile center
DISTANCE_WEIGHT = 1

TILE_LEVEL_FROM_ROOT = 4
NUM_TILES_PER_FRAME = 8**TILE_LEVEL_FROM_ROOT

# side length in num_tiles
NUM_TILES_PER_SIDE_IN_A_FRAME = 2**TILE_LEVEL_FROM_ROOT

NUM_FRAMES = 300

VIDEO_LENGTH = 60  # sec

# video: H1 user: P01_V1
# NUM_FRAMES_VIEWED = 549 + BUFFER_LENGTH # watching looply
NUM_FRAMES_VIEWED = VIDEO_LENGTH * FPS  # watching looply
NUM_UPDATES = NUM_FRAMES_VIEWED // FPS
# NUM_UPDATES = 3

VIDEO_NAME = 'longdress'
VALID_TILES_PATH = '../valid_tiles/' + VIDEO_NAME

# only have longdress data
VALID_TILES_PATH_FROM_NUM_PTS = '../psnr_weights/results_map_tileRate_to_lod/rate_versions_6x300x16x16x16.pkl'

if VIDEO_NAME == 'longdress':
    fov_traces_file = 'H1_nav.csv'
elif VIDEO_NAME == 'loot':
    fov_traces_file = 'H2_nav.csv'
elif VIDEO_NAME == 'readandblack':
    fov_traces_file = 'H3_nav.csv'
elif VIDEO_NAME == 'soldier':
    fov_traces_file = 'H4_nav.csv'
else:
    pass
# FOV_TRACES_PATH = '../fov_traces/6DoF-HMD-UserNavigationData/NavigationData/' + fov_traces_file
FOV_TRACES_PATH = '../fov_traces/6DoF-HMD-UserNavigationData/NavigationData/H1_u7.txt'

BANDWIDTH_TRACES_PATH = '../bw_traces/100ms_loss1.txt'
# BANDWIDTH_TRACES_PATH = '../bw_traces/square_wave.txt'
# BANDWIDTH_TRACES_PATH = '../bw_traces/constant_wave.txt'
# BANDWIDTH_TRACES_PATH = '../bandwidth_5G/BW_Trace_5G_0.txt'

FOV_PREDICTION_HISTORY_WIN_LENGTH = (
    BUFFER_LENGTH + UPDATE_FREQ * FPS
) // 2  # frame; according to vivo paper, best use half of prediciton window
OVERLAP_RATIO_HISTORY_WIN_LENGTH = 10
BW_PREDICTION_HISTORY_WIN_LENGTH = 5  # in second

OBJECT_SIDE_LEN = 1.8  # meter, according to http://plenodb.jpeg.org/pc/8ilabs
TILE_SIDE_LEN = OBJECT_SIDE_LEN / NUM_TILES_PER_SIDE_IN_A_FRAME  # meter

SATURATION_RESOLUTION = 60 # angular resolution, i.e. # points per degree
QR_MODEL_LOG_FACTOR = np.exp(1) / SATURATION_RESOLUTION # Q(r)=theta * log(c * resol), where c is log factor
#THETA_FACTOR = 0.01

FOV_DEGREE_SPAN = np.pi / 2  # 90 degrees, a circle fov

QR_WEIGHTS_PATH_A = '../psnr_weights/results_map_tileRate_to_lod/new_a_300x16x16x16.pkl'
QR_WEIGHTS_PATH_B = '../psnr_weights/results_map_tileRate_to_lod/new_b_300x16x16x16.pkl'
PATH_RATE_VERSIONS = '../psnr_weights/results_map_tileRate_to_lod/rate_versions_6x300x16x16x16.pkl'
PATH_NUM_PTS_VERSIONS = '../psnr_weights/results_map_tileRate_to_lod/num_pts_versions_6x300x16x16x16.pkl'

BANDWIDTH_ORACLE_KNOWN = True
if MMSYS_HYBRID_TILING:
    BANDWIDTH_ORACLE_KNOWN = True

FOV_ORACLE_KNOW = False

Mbps_TO_Bps = 1e6 / 8

# MAP_6DOF_TO_HMD_DATA = {'x':'HMDPX', 'y':'HMDPY', 'z':'HMDPZ', 'pitch':'HMDRX', 'yaw':'HMDRY', 'roll':'HMDRZ'}
MAP_6DOF_TO_HMD_DATA = {
    'x': 0,
    'y': 1,
    'z': 2,
    'pitch': 3,
    'yaw': 4,
    'roll': 5
}

MAP_DOF_TO_PLOT_POS = {
    'x': [0, 0],
    'y': [0, 1],
    'z': [0, 2],
    'pitch': [1, 0],
    'yaw': [1, 1],
    'roll': [1, 2]
}

QUANTIZE_TILE_SIZE = False

ROUND_TILE_SIZE = False

SCALE_BW = 5  # know bw oracle, otherwise not correct

MAP_VERSION_TO_SIZE = {
    0: 0,
    1: BYTE_SIZES[0],
    2: BYTE_SIZES[1],
    3: BYTE_SIZES[2]
}
MAP_SIZE_TO_VERSION = {
    0: 0,
    BYTE_SIZES[0]: 1,
    BYTE_SIZES[1]: 2,
    BYTE_SIZES[2]: 3
}

NUM_RATE_VERSIONS = 6
BARRIER_WEIGHT = 1e-9
QUALITY_SUM_WEIGHT = 500

PROGRESSIVE_DOWNLOADING = True
FAKE_MAE_ERROR = False
TRUNCATE_LINEAR_REGRESSION = True
SCALE_FOV_PREDICTION_WIN = 0.5
FIX_FOV_PRED_WIN = False
FOV_PRED_WIN_LEN = 1
