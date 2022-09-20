import cv2
import numpy as np
import torch

from model.OTPose2 import ViSegTransPoseH
from utils.bbox import box2cs
from utils.heatmap import get_final_preds
from utils.images import read_image
from utils.setup import get_cfg, update_config
from utils.transform import build_transforms, get_affine_transform
import os
import sys
import argparse

root_dir = os.path.abspath('../')
INFERENCE_PHASE = "inference"


def parse_args():
    parser = argparse.ArgumentParser(description='Inference pose estimation Network')

    parser.add_argument('--cfg', help='experiment configure file name', required=False, type=str,
                        default="./configs/17/model_RSN_inference.yaml")
    parser.add_argument('--PE_Name', help='pose estimation model name', required=False, type=str,
                        default='OTPose')
    parser.add_argument('-weight', help='model weight file', required=False, type=str
                        , default='./output/PE/SemiCTT/HAViT_cos_fix_nf/PoseTrack18/checkpoints//best_mAP_83.91406382334866_state.pth')
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    # philly
    args = parser.parse_args()
    args.rootDir = root_dir
    args.cfg = os.path.abspath(os.path.join(args.rootDir, args.cfg))
    args.weight = os.path.abspath(os.path.join(args.rootDir, args.weight))
    print(args.weight)
    return args


def get_inference_model():
    global cfg, args
    args = parse_args()
    cfg = get_cfg(args)
    update_config(cfg, args)
    checkpoint_dict = torch.load(args.weight)
    model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
    new_model = ViSegTransPoseH(cfg, phase=INFERENCE_PHASE)
    new_model.load_state_dict(model_state_dict)
    return new_model.cuda()


model = get_inference_model()
image_transforms = build_transforms(None, INFERENCE_PHASE)
image_size = np.array([288, 384])
aspect_ratio = image_size[0] * 1.0 / image_size[1]


def image_preprocess(image_path: str, prev_image: str, next_image: str, pprev_image: str, nnext_image: str, center, scale):
    trans_matrix = get_affine_transform(center, scale, 0, image_size)
    image_data = read_image(image_path)
    image_data = cv2.warpAffine(image_data, trans_matrix, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
    image_data = image_transforms(image_data)
    if prev_image is None or next_image is None:
        return image_data
    else:
        prev_image_data = read_image(prev_image)
        prev_image_data = cv2.warpAffine(prev_image_data, trans_matrix, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        prev_image_data = image_transforms(prev_image_data)

        pprev_image_data = read_image(pprev_image)
        pprev_image_data = cv2.warpAffine(pprev_image_data, trans_matrix, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        pprev_image_data = image_transforms(pprev_image_data)

        next_image_data = read_image(next_image)
        next_image_data = cv2.warpAffine(next_image_data, trans_matrix, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        next_image_data = image_transforms(next_image_data)

        nnext_image_data = read_image(nnext_image)
        nnext_image_data = cv2.warpAffine(nnext_image_data, trans_matrix, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
        nnext_image_data = image_transforms(nnext_image_data)

        return image_data, prev_image_data, next_image_data, pprev_image_data, nnext_image_data

def inference_PE(input_image: str, prev_image: str, next_image: str, pprev_image: str, nnext_image: str, bbox):
    """
        input_image : input image path
        prev_image : prev image path
        next_image : next image path
        inference pose estimation
    """
    center, scale = box2cs(bbox, aspect_ratio)
    target_image_data, prev_image_data, next_image_data, pprev_image_data, nnext_image_data = image_preprocess(input_image, prev_image, next_image, pprev_image, nnext_image, center, scale)

    target_image_data = target_image_data.unsqueeze(0)
    prev_image_data = prev_image_data.unsqueeze(0)
    next_image_data = next_image_data.unsqueeze(0)
    pprev_image_data = pprev_image_data.unsqueeze(0)
    nnext_image_data = nnext_image_data.unsqueeze(0)

    concat_input = torch.cat((target_image_data, prev_image_data, next_image_data,
                              pprev_image_data, nnext_image_data), 1).cuda()
    margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0),
                          torch.tensor(2).unsqueeze(0), torch.tensor(2).unsqueeze(0)], dim=1).cuda()
    model.eval()

    predictions = model(concat_input, margin=margin)

    pred_joint, pred_conf = get_final_preds(predictions[0].cpu().detach().numpy(), [center], [scale])
    pred_keypoints = np.concatenate([pred_joint.astype(int), pred_conf], axis=2)

    return pred_keypoints