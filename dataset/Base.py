import logging
from tabulate import tabulate
import numpy as np

from utils.setup import get_dataset_name


class BaseDataset:
    def __init__(self, cfg, phase, **kwargs):
        self.phase = phase
        self.dataset_name = get_dataset_name(cfg)
        self.pixel_std = 200

        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.image_width = self.image_size[0]
        self.image_height = self.image_size[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)

        self.scale_factor = cfg.TRAIN.SCALE_FACTOR
        self.rotation_factor = cfg.TRAIN.ROT_FACTOR
        self.flip = cfg.TRAIN.FLIP

        self.color_rgb = cfg.DATASET.COLOR_RGB
        self.num_joints_half_body = cfg.TRAIN.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.TRAIN.PROB_HALF_BODY
        self.num_joints = cfg.MODEL.NUM_JOINTS

        # Loss
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT

        self.data = []

    def __len__(self):
        return len(self.data)

    def show_samples(self):
        logger = logging.getLogger(__name__)
        table_header = ["Dataset_Name", "Num of samples"]
        table_data = [[self.dataset_name, len(self.data)]]

        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Datasets Samples Info : \n" + table)

    def show_data_parameters(self):
        logger = logging.getLogger(__name__)
        table_header = ["Dataset parameters", "Value"]
        table_data = [
            ["BBOX_ENLARGE_FACTOR", self.bbox_enlarge_factor],
            ["NUM_JOINTS", self.num_joints]
        ]
        if self.phase != 'train':
            table_extend_data = [
                []
            ]
            table_data.extend(table_extend_data)
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Datasets Parameters Info : \n" + table)