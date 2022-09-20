from random import random
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def box2cs(box, aspect_ratio, enlarge_factor=1.0):
    """
        box( x y w h ) convert to center and scale

        x,y is top left corner
    """
    x, y, w, h = box[:4]
    return xywh2cs(x, y, w, h, aspect_ratio, enlarge_factor)


def xywh2cs(x, y, w, h, aspect_ratio, enlarge_factor):
    center = np.zeros(2, dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    pixel_std = 200
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * enlarge_factor

    return center, scale


def cs2box(center, scale, pixel_std=200, pattern="xywh"):
    """
        center, scale convert to bounding box
        pattern in ["xywh","xyxy"] . default: "xywh"
            xywh - xy upper left corner of bbox , w h is width and height of bbox respectively
            xyxy - upper left corner and bottom right corner
    """
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std

    if pattern == "xyxy":
        # "xyxy" pattern
        x1 = center[0] - w * 0.5
        y1 = center[1] - h * 0.5
        x2 = center[0] + w * 0.5
        y2 = center[1] + h * 0.5
        return [x1, y1, x2, y2]
    else:
        # "xywh" pattern
        x = center[0] - w * 0.5
        y = center[1] - h * 0.5
        return [x, y, w, h]


def add_bbox_in_image(image, bbox, color=None, label=None, line_thickness=None, multi_language: bool = False):
    """
    :param image
    :param bbox   -  xyxy
    :param color
    :param label
    :param line_thickness
    :param multi_language
    """
    if color is None:
        color = (random() * 255, random() * 255, random() * 255)

    if line_thickness is None:
        line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

    x1, y1, x2, y2 = map(int, bbox)

    corner1 = (x1, y1)
    corner2 = (x2, y2)

    image_with_bbox = cv2.rectangle(image, corner1, corner2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    if label:
        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=font_thickness / 3, thickness=font_thickness)[0]
        text_corner2 = corner1[0] + t_size[0], corner1[1] - t_size[1] - 3
        cv2.rectangle(image, corner1, text_corner2, -1, cv2.LINE_AA)  # filled
        if not multi_language:
            cv2.putText(image, label, (corner1[0], corner1[1] - 2), 0, font_thickness / 3, [225, 255, 255], thickness=font_thickness,
                        lineType=cv2.LINE_AA)
        else:
            font_path = "font/simsun.ttc"
            font = ImageFont.truetype(font_path, 64)
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)
            draw.text((corner1[0], corner1[1] - 2), label, font=font, fill=(225, 255, 255))
    return image_with_bbox
