import sys
from collections import defaultdict

from tabulate import tabulate

from configs.arg_parse import default_parse_args
from utils.setup import create_folder, get_dataset_name, setup
import os.path as osp
import logging
import time

class Base:
    def __init__(self, phase: str):
        self._hooks = []
        self.output_path_dict = {}
        # self.cfg = cfg
        self.phase = phase
        self.checkpoints_save_folder = None
        self.tb_save_folder = None
        self.log_file = None

        self.args = default_parse_args()
        self.cfg = setup(self.args)
        self.output_path_dict = defaultdict(str)

        self.setup_cfg()

        self.update_output_paths(self.output_path_dict, phase)

    def setup_cfg(self):
        cfg = self.cfg
        dataset_name = get_dataset_name(self.cfg)
        cfg.defrost()
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, cfg.EXPERIMENT_NAME, dataset_name)
        cfg.freeze()

        checkpoints_save_folder = osp.join(cfg.OUTPUT_DIR, "checkpoints")
        log_save_folder = osp.join(cfg.OUTPUT_DIR, "log")
        writer_save_folder = osp.join(cfg.OUTPUT_DIR, 'tensorboard')
        create_folder(checkpoints_save_folder)
        create_folder(writer_save_folder)
        create_folder(log_save_folder)

        self.output_path_dict["checkpoints_save_folder"] = checkpoints_save_folder
        self.output_path_dict["tb_save_folder"] = writer_save_folder  # tensorboard save
        self.output_path_dict["log_save_folder"] = log_save_folder

    def update_output_paths(self, output_paths, phase):
        log_save_folder = output_paths.get("log_save_folder", "./log")
        create_folder(log_save_folder)
        log_file = osp.join(log_save_folder, "{}-{}.log".format(phase, time.strftime("%Y_%m_%d_%H")))

        self.checkpoints_save_folder = output_paths["checkpoints_save_folder"]
        self.tb_save_folder = output_paths["tb_save_folder"]
        self.log_file = log_file

        self.reset_logger(self.log_file)

        self.show_info()

    def show_info(self):
        logger = logging.getLogger(__name__)
        table_header = ["Key", "Value"]
        table_data = [
            ["Phase", self.phase],
            ["Log File", self.log_file],
            ["Checkpoint Folder", self.checkpoints_save_folder],
            ["Tensorboard_save_folder", self.tb_save_folder],
        ]
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Executor Operating Parameter Table: \n" + table)

    # setup root logger
    def setup_logger(self, save_file, logger=None, logger_level=logging.DEBUG, **kwargs):
        if logger is None:
            logger = logging.getLogger()
        logger.setLevel(logger_level)
        # log_file
        file_handler = logging.FileHandler(save_file)
        file_handler.setLevel(logger_level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))

    def reset_logger(self, save_file, logger=None, logger_level=logging.DEBUG, **kwargs):
        local_rank = kwargs.get("local_rank", -1)

        if local_rank <= 0:
            if logger is None:
                logger = logging.getLogger()
            while logger.hasHandlers():
                logger.removeHandler(logger.handlers[0])
            self.setup_logger(save_file, logger, logger_level)
