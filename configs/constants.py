#  PoseTrack Official Keypoint Ordering  - A total of 15 2017
PoseTrack_Official_Keypoint_Ordering = [
    'right_ankle',      # 0
    'right_knee',       # 1
    'right_hip',        # 2
    'left_hip',         # 3
    'left_knee',        # 4
    'left_ankle',       # 5
    'right_wrist',      # 6
    'right_elbow',      # 7
    'right_shoulder',   # 8
    'left_shoulder',    # 9
    'left_elbow',       # 10
    'left_wrist',
    'head_bottom',      # 12
    'nose',
    'head_top',         # 14
]

# Endpoint1 , Endpoint2 , line_color 2017
PoseTrack_Official_Keypoint_Pairs = [
    [14, 12, 'Rosy'],   # ['head_top', 'head_bottom', 'Rosy'],
    [12, 8, 'Yellow'],  # ['head_bottom', 'right_shoulder', 'Yellow'],
    [12, 9, 'Yellow'],  # ['head_bottom', 'left_shoulder', 'Yellow'],
    [8, 7, 'Blue'],     # ['right_shoulder', 'right_elbow', 'Blue'],
    [7, 6, 'Blue'],     # ['right_elbow', 'right_wrist', 'Blue'],
    [9, 10, 'Green'],   # ['left_shoulder', 'left_elbow', 'Green'],
    [10, 11, 'Green'],  # ['left_elbow', 'left_wrist', 'Green'],
    [8, 2, 'Purple'],   # ['right_shoulder', 'right_hip', 'Purple'],
    [9, 3, 'SkyBlue'],  # ['left_shoulder', 'left_hip', 'SkyBlue'],
    [2, 1, 'Purple'],   # ['right_hip', 'right_knee', 'Purple'],
    [1, 0, 'Purple'],   # ['right_knee', 'right_ankle', 'Purple'],
    [3, 4, 'SkyBlue'],  # ['left_hip', 'left_knee', 'SkyBlue'],
    [4, 5, 'SkyBlue'],  # ['left_knee', 'left_ankle', 'SkyBlue'],
]

# Facebook PoseTrack Keypoint Ordering (convert to COCO format)  -   A total of 17 2018
PoseTrack_COCO_Keypoint_Ordering = [
    'nose',                 # 0
    'head_bottom',          # 1
    'head_top',             # 2
    'left_ear',             # 3
    'right_ear',            # 4
    'left_shoulder',        # 5
    'right_shoulder',       # 6
    'left_elbow',           # 7
    'right_elbow',          # 8
    'left_wrist',           # 9
    'right_wrist',          # 10
    'left_hip',             # 11
    'right_hip',            # 12
    'left_knee',            # 13
    'right_knee',           # 14
    'left_ankle',           # 15
    'right_ankle',          # 16
]

# Endpoint1 , Endpoint2 , line_color 2018
PoseTrack_Keypoint_Pairs = [
    # [2, 0, 'Rosy'],         # ['head_top', 'nose', 'Rosy'],
    # [0, 1, 'Rosy'],         # ['nose', 'head_bottom', 'Rosy'],
    [2, 1, 'Rosy'],  # ['head_top', 'head_bottom', 'Rosy'],
    # [0, 3, 'Rosy'],
    # [0, 4, 'Rosy'],
    [1, 6, 'Yellow'],       # ['head_bottom', 'right_shoulder', 'Yellow'],
    [1, 5, 'Yellow'],       # ['head_bottom', 'left_shoulder', 'Yellow'],
    [6, 8, 'Blue'],         # ['right_shoulder', 'right_elbow', 'Blue'],
    [8, 10, 'Blue'],         # ['right_elbow', 'right_wrist', 'Blue'],
    [5, 7, 'Green'],        # ['left_shoulder', 'left_elbow', 'Green'],
    [7, 9, 'Green'],        # ['left_elbow', 'left_wrist', 'Green'],
    [6, 12, 'Purple'],      # ['right_shoulder', 'right_hip', 'Purple'],
    [5, 11, 'SkyBlue'],     # ['left_shoulder', 'left_hip', 'SkyBlue'],
    # [11, 12, 'Yellow'],
    [12, 14, 'Purple'],     # ['right_hip', 'right_knee', 'Purple'],
    [14, 16, 'Purple'],     # ['right_knee', 'right_ankle', 'Purple'],
    [11, 13, 'SkyBlue'],    # ['left_hip', 'left_knee', 'SkyBlue'],
    [13, 15, 'SkyBlue'],    # ['left_knee', 'left_ankle', 'SkyBlue'],
]


PoseTrack_Keypoint_Name_Colors = [['right_ankle', 'Gold'],
                                  ['right_knee', 'Orange'],
                                  ['right_hip', 'DarkOrange'],
                                  ['left_hip', 'Peru'],
                                  ['left_knee', 'LightSalmon'],
                                  ['left_ankle', 'OrangeRed'],
                                  ['right_wrist', 'LightGreen'],
                                  ['right_elbow', 'LimeGreen'],
                                  ['right_shoulder', 'ForestGreen'],
                                  ['left_shoulder', 'DarkTurquoise'],
                                  ['left_elbow', 'Cyan'],
                                  ['left_wrist', 'PaleTurquoise'],
                                  ['head_bottom', 'DoderBlue'],
                                  ['nose', 'HotPink'],
                                  ['head_top', 'SlateBlue']]

COLOR_DICT = {
    'Rosy': (255, 47, 130),
    'Purple': (252, 176, 243),
    'Yellow': (255, 156, 49),
    'Blue': (107, 183, 190),
    'Green': (76, 255, 160),
    'SkyBlue': (76, 288, 255),
    'HotPink': (255, 105, 180),
    'SlateBlue': (106, 90, 205),
    'DoderBlue': (30, 144, 255),
    'PaleTurquoise': (175, 238, 238),
    'Cyan': (0, 255, 255),
    'DarkTurquoise': (0, 206, 209),
    'ForestGreen': (34, 139, 34),
    'LimeGreen': (50, 205, 50),
    'LightGreen': (144, 238, 144),
    'OrangeRed': (255, 69, 0),
    'Orange': (255, 165, 0),
    'LightSalmon': (255, 160, 122),
    'Peru': (205, 133, 63),
    'DarkOrange': (255, 140, 0),
    'Gold': (255, 215, 0),
}
