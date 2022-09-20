import torch
import cv2
import numpy as np
import os.path as osp

from configs.constants import PoseTrack_Keypoint_Pairs, PoseTrack_Official_Keypoint_Ordering, COLOR_DICT, \
    PoseTrack_Official_Keypoint_Pairs
from utils.bbox import add_bbox_in_image
from utils.keypoints import coco2posetrack_ord_infer
from utils.setup import create_folder, list_immediate_childfile_paths
from utils.transform import mean, std
import os

def tensor2im(input_image, imtype=np.uint8):
    """"
        tensor -> numpy , and normalize

    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  #
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        # (BGR)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        # (channels, height, width) to (height, width, channels)

        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def draw_skeleton_in_origin_image(batch_image_list, batch_joints_list, batch_bbox_list, save_dir, vis_skeleton=True, vis_bbox=True):
    """
    :param batch_image_list:  batch image path
    :param batch_joints_list:   joints coordinates in image Coordinate reference system
    :batch_bbox_list: xyxy
    :param save_dir:
    :return: No return
    """

    skeleton_image_save_folder = osp.join(save_dir, "skeleton")
    bbox_image_save_folder = osp.join(save_dir, "bbox")
    together_save_folder = osp.join(save_dir, "SkeletonAndBbox")

    if vis_skeleton and vis_bbox:
        save_folder = together_save_folder
    else:
        save_folder = skeleton_image_save_folder
        if vis_bbox:
            save_folder = bbox_image_save_folder

    batch_final_coords = batch_joints_list

    for index, image_path in enumerate(batch_image_list):
        final_coords = batch_final_coords[index]
        final_coords = coco2posetrack_ord_infer(final_coords)
        bbox = batch_bbox_list[index]

        image_name = image_path[image_path.index("images") + len("images") + 1:]
        # image_name = image_path[image_path.index("frames") + len("frames") + 1:]

        vis_image_save_path = osp.join(save_folder, image_name)
        if osp.exists(vis_image_save_path):
            processed_image = cv2.imread(vis_image_save_path)

            processed_image = add_poseTrack_joint_connection_to_image(processed_image, final_coords, sure_threshold=0.2,
                                                                      flag_only_draw_sure=True) if vis_skeleton else processed_image
            processed_image = add_bbox_in_image(processed_image, bbox) if vis_bbox else processed_image

            save_image(vis_image_save_path, processed_image)
        else:
            image_data = cv2.imread(image_path)
            processed_image = image_data.copy()

            processed_image = add_poseTrack_joint_connection_to_image(processed_image, final_coords, sure_threshold=0.2,
                                                                      flag_only_draw_sure=True) if vis_skeleton else processed_image

            processed_image = add_bbox_in_image(processed_image, bbox) if vis_bbox else processed_image

            save_image(vis_image_save_path, processed_image)


def add_poseTrack_joint_connection_to_image(img_demo, joints, sure_threshold=0.8, flag_only_draw_sure=False, ):
    for joint_pair in PoseTrack_Official_Keypoint_Pairs:
        ind_1 = joint_pair[0]
        ind_2 = joint_pair[1]

        color = COLOR_DICT[joint_pair[2]]

        x1, y1, sure1 = joints[ind_1]
        x2, y2, sure2 = joints[ind_2]

        if x1 <= 5 and y1 <= 5: continue
        if x2 <= 5 and y2 <= 5: continue

        if flag_only_draw_sure is False:
            sure1 = sure2 = 1
        if sure1 > sure_threshold and sure2 > sure_threshold:
            # if sure1 > 0.8 and sure2 > 0.8:
            # cv2.line(img_demo, (x1, y1), (x2, y2), color, thickness=8)
            cv2.line(img_demo, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=6)
    return img_demo


def circle_vis_point(img, joints):
    for joint in joints:
        x, y, c = [int(i) for i in joint]
        cv2.circle(img, (x, y), 3, (255, 255, 255), thickness=3)

    return img


def save_image(image_save_path, image_data):
    create_folder(osp.dirname(image_save_path))
    return cv2.imwrite(image_save_path, image_data, [100])


def read_image(image_path):
    if not osp.exists(image_path):
        raise Exception("Failed to read image from path : {}".format(image_path))
    img = cv2.imread(image_path)

    return img

def folder_exists(folder_path):
    return os.path.exists(folder_path)

def video2images(video_path, outimages_path=None, zero_fill=8):
    cap = cv2.VideoCapture(video_path)
    isOpened = cap.isOpened()
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    i = 0
    if outimages_path is not None:
        if not folder_exists(outimages_path):
            create_folder(outimages_path)
    assert isOpened, "Can't find video"
    for index in range(video_length):
        (flag, data) = cap.read()
        file_name = "{}.jpg".format(str(index).zfill(zero_fill))  # start from zero
        if outimages_path is not None:
            file_path = osp.join(outimages_path, file_name)
        else:
            create_folder("output")
            file_path = osp.join("output", file_name)
        if flag:

            cv2.imwrite(file_path, data, [cv2.IMWRITE_JPEG_QUALITY, 100])


def image2video(image_dir, name, fps=25):
    image_path_list = []
    for image_path in list_immediate_childfile_paths(image_dir):
        image_path_list.append(image_path)
    image_path_list.sort()
    temp = cv2.imread(image_path_list[0])
    size = (temp.shape[1], temp.shape[0])
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter('./output/' + name + '.mp4', fourcc, fps, size)
    for image_path in image_path_list:
        if image_path.endswith(".jpg"):
            image_data_temp = cv2.imread(image_path)
            video.write(image_data_temp)
    print("Video done！")