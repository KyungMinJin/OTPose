import numpy as np
import math

import torch
from PIL import Image
from matplotlib import pyplot as plt

from utils.transform import get_affine_transform, exec_affine_transform


def generate_pred_heatmaps(joints, sigma, image_size, heatmap_size, num_joints):
    target = np.zeros((num_joints,
                       heatmap_size[1],
                       heatmap_size[0]),
                      dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target


def generate_heatmaps(joints, joints_vis, sigma, image_size, heatmap_size, num_joints, **kwargs):
    """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :param sigma:
        :param image_size:
        :param heatmap_size:
        :param num_joints:
        :return: target, target_weight(1: visible, 0: invisible)
    """

    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((num_joints,
                       heatmap_size[1],
                       heatmap_size[0]),
                      dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    if ("use_different_joints_weight" in kwargs) and (kwargs["use_different_joints_weight"]):
        target_weight = np.multiply(target_weight, kwargs["joints_weight"])

    return target, target_weight


def get_final_preds(batch_heatmaps, center, scale):  # heatmap [batch,channel,width,height]
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]])
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):  # batch
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = exec_affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    # assert isinstance(batch_heatmaps, np.ndarray), \
    #     'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def normalize_0_to_1(heatmaps):
    min_val = heatmaps.min(-1)[0].min(-1)[0]
    max_val = heatmaps.max(-1)[0].max(-1)[0]
    heatmaps = (heatmaps - min_val[:, :, None, None]) / (max_val[:, :, None, None])
    return heatmaps


def adjust_sigma(epoch, sigma, schedule, gamma=0.5):
    """Sets the learning rate to the initial LR decayed by schedule"""
    for step in schedule:
        if epoch >= step:
            sigma -= 1

    return max(sigma, 1)

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

#
# def plot_seg(d_seg, seg_x, raw_img):
#     palette = get_palette(d_seg)
#     # for i, seg_img in enumerate(seg_x[0][-1]):
#     # upsample_output = seg_x[0][-1][0]
#     upsample = torch.nn.Upsample(size=[raw_img.shape[3], raw_img.shape[2]], mode='bilinear', align_corners=True)
#     upsample_output = upsample(seg_x[0][-1][0].unsqueeze(0))
#     upsample_output = upsample_output.squeeze()
#     upsample_output = upsample_output.permute(1, 2, 0) # CHW -> HWC
#     logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=[raw_img.shape[3], raw_img.shape[2]])
#     parsing_result = np.argmax(upsample_output.permute(1, 2, 0).clone().detach().cpu().numpy(), axis=2)  # h, w, c
#     # seg_img = Image.fromarray(np.asarray(seg_x[0][0], dtype=np.uint8))
#     seg_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
#     seg_img.putpalette(palette)
#
#     fig = plt.figure()
#     rows = 1
#     cols = 2
#     ax1 = fig.add_subplot(rows, cols, 1)
#     ax1.imshow(seg_img)
#     ax1.set_title('Segmentation')
#     ax2 = fig.add_subplot(rows, cols, 2)
#     ax2.imshow(raw_img[0].permute(1,2,0).cpu().numpy())
#     ax2.set_title('raw img')
#
#     plt.show()