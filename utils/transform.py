import torchvision.transforms as T
import numpy as np
import cv2
import torch

# RGB
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def build_transforms(cfg, phase):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return transform


def half_body_transform(joints, joints_vis, num_joints, upper_body_ids, aspect_ratio, pixel_std):
    upper_joints = []
    lower_joints = []
    for joint_id in range(num_joints):
        if joints_vis[joint_id][0] > 0:
            if joint_id in upper_body_ids:
                upper_joints.append(joints[joint_id])
            else:
                lower_joints.append(joints[joint_id])

    if np.random.randn() < 0.5 and len(upper_joints) > 2:
        selected_joints = upper_joints
    else:
        selected_joints = lower_joints if len(lower_joints) > 2 else upper_joints

    if len(selected_joints) < 2:
        return None, None

    selected_joints = np.array(selected_joints, dtype=np.float32)
    center = selected_joints.mean(axis=0)[:2]

    left_top = np.amin(selected_joints, axis=0)
    right_bottom = np.amax(selected_joints, axis=0)

    w = right_bottom[0] - left_top[0]
    h = right_bottom[1] - left_top[1]

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio  # aspect_ratio=w/h
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)

    scale = scale * 1.5

    return center, scale


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints * joints_vis, joints_vis


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def exec_affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def reverse_transforms(batch_tensor: torch.Tensor):
    """
    tensor
    """
    if batch_tensor.shape[1] == 1:  # grayscale to RGB
        batch_tensor = batch_tensor.repeat((1, 3, 1, 1))

    for i in range(len(mean)):
        batch_tensor[:, i, :, :] = batch_tensor[:, i, :, :] * std[i] + mean[i]

    batch_tensor = batch_tensor * 255
    # RGB -> BGR
    RGB_batch_tensor = batch_tensor.split(1, dim=1)
    batch_tensor = torch.cat([RGB_batch_tensor[2], RGB_batch_tensor[1], RGB_batch_tensor[0]], dim=1)
    return batch_tensor
