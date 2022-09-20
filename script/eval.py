import logging
import time
from collections import defaultdict
import os.path as osp
import torch
from tensorboardX import SummaryWriter

from configs.arg_parse import default_parse_args
from dataset.PoseTrackDataset import PoseTrackDataset
from model.simple import HRNet
from script.Common import CommonFunction
from script.base import Base
from utils.setup import create_folder, get_latest_checkpoint, get_all_checkpoints


class Eval(Base):
    def __init__(self, phase='validate', **kwargs):
        super().__init__()
        cfg = self.cfg
        args = self.args

        self.phase = phase
        self.PE_Name = args.PE_Name
        self.setup_cfg()

        self.checkpoints_save_folder = self.output_path_dict["checkpoints_save_folder"]
        self.tb_save_folder = self.output_path_dict["tb_save_folder"]

        dataset = PoseTrackDataset(cfg=cfg, phase=phase)
        if phase == 'validate':
            batch_size = cfg.VAL.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
        elif phase == 'test':
            batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
        else:
            raise BaseException

        eval_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

        self.dataloader = eval_loader
        self.model = HRNet.get_net(cfg, phase=phase)
        self.model.eval()
        self.dataset = self.dataloader.dataset
        self.GPUS = cfg.GPUS
        self.output = cfg.OUTPUT_DIR

        logger = logging.getLogger(__name__)
        log_save_folder = self.output_path_dict.get("log_save_folder", "./log")
        create_folder(log_save_folder)
        log_file = osp.join(log_save_folder, "{}-{}.log".format('train', time.strftime("%Y_%m_%d_%H")))
        self.log_file = log_file

        self.eval_from_checkpoint_id = kwargs.get("eval_from_checkpoint_id", -1)
        self.evaluate_model_state_files = []
        self.list_evaluate_model_files(cfg, phase)
        self.core_function = CommonFunction(cfg)
        self.tb_writer_dict = {"writer": SummaryWriter(self.tb_save_folder),
                               "global_steps": 0}

        self.eval()

    def list_evaluate_model_files(self, cfg, phase):
        subCfgNode = cfg.VAL if phase == 'validate' else cfg.TEST
        if subCfgNode.MODEL_FILE:
            if subCfgNode.MODEL_FILE[0] == '.':
                model_state_file = osp.abspath(osp.join(self.checkpoints_save_folder, subCfgNode.MODEL_FILE))
            else:
                model_state_file = osp.join(self.cfg.ROOT_DIR, subCfgNode.MODEL_FILE)

            # model_state_file = osp.abspath(osp.join(cfg.ROOT_DIR, subCfgNode.MODEL_FILE))
            self.evaluate_model_state_files.append(model_state_file)
        else:
            if self.eval_from_checkpoint_id == -1:
                model_state_file = get_latest_checkpoint(self.checkpoints_save_folder)
                self.evaluate_model_state_files.append(model_state_file)
            else:
                candidate_model_files = get_all_checkpoints(self.checkpoints_save_folder)
                for model_file in candidate_model_files:
                    model_file_epoch_num = int(osp.basename(model_file).split("_")[1])
                    if model_file_epoch_num >= self.eval_from_checkpoint_id:
                        self.evaluate_model_state_files.append(model_file)

    def eval(self):
        if len(self.evaluate_model_state_files) == 0:
            logger = logging.getLogger(__name__)
            logger.error("=> No model state file available for evaluation")
        else:
            for model_checkpoint_file in self.evaluate_model_state_files:
                model, epoch = self.model_load(model_checkpoint_file)
                self.core_function.eval(model=model, dataloader=self.dataloader, tb_writer_dict=self.tb_writer_dict, epoch=epoch,
                                        phase=self.phase)

    def model_load(self, checkpoints_file):
        logger = logging.getLogger(__name__)
        logger.info("=> loading checkpoints from {}".format(checkpoints_file))
        checkpoint_dict = torch.load(checkpoints_file)
        epoch = checkpoint_dict.get("begin_epoch", "0")
        model = self.model

        if "state_dict" in checkpoint_dict:
            model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}

        else:
            model_state_dict = checkpoint_dict

        if self.PE_Name == 'MSRA':
            model_state_dict = {k.replace('rough_pose_estimation_net.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        if len(self.GPUS) > 1:
            model = torch.nn.DataParallel(model.cuda())
        else:
            model = model.cuda()
        return model, epoch


if __name__ == '__main__':
    Eval(phase='validate')
