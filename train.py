import logging
import os
import torch
from tensorboardX import SummaryWriter
from dataset.PoseTrackDataset import PoseTrackDataset
from eval import Eval
from model.OTPose import OTPose
from model.checkpoints import resume, save_checkpoint, save_best_checkpoint
from model.loss import build_loss
from script.Common import CommonFunction
from script.base import Base
from thirdparty.utils import make_optimizer, make_scheduler, fix_random_seed
from thirdparty.utils.data_utils import worker_init_reset_seed
from utils.heatmap import adjust_sigma
from utils.setup import get_latest_checkpoint
from utils.model_env import set_random_seed


class Train(Base):
    def __init__(self):
        super().__init__(phase="train")
        cfg = self.cfg
        args = self.args
        logger = logging.getLogger(__name__)

        self.is_train = args.train
        self.val = args.val
        self.test = args.test
        self.val_from_checkpoint_id = args.val_from_checkpoint
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((map(str, cfg.GPUS)))
        self.gpus = cfg.GPUS
        self.PE_Name = args.PE_Name

        self.checkpoints_save_folder = self.output_path_dict["checkpoints_save_folder"]
        self.tb_save_folder = self.output_path_dict["tb_save_folder"]

        logger.info("Set the random seed to {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

        dataset = PoseTrackDataset(cfg=cfg, phase='train')

        batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=cfg.WORKERS,
            worker_init_fn=(worker_init_reset_seed),
            shuffle=cfg.TRAIN.SHUFFLE,
            drop_last=True,
            pin_memory=cfg.PIN_MEMORY,
        )

        self.dataloader = train_loader

        # model
        self.model = OTPose(cfg)
        self.model.train()
        self.optimizer = make_optimizer(self.model, cfg)
        self.lr_scheduler = make_scheduler(self.optimizer, cfg, num_iters_per_epoch=len(train_loader))
        self.loss_criterion = build_loss(cfg)
        self.begin_epoch = 0
        self.end_epoch = cfg.TRAIN.END_EPOCH + cfg.TRAIN.WARMUP_EPOCHS
        self.save_model_per_epoch = cfg.TRAIN.SAVE_MODEL_PER_EPOCH

        self.model = self.model.cuda()
        self.GPUS = cfg.GPUS
        self.core_function = CommonFunction(cfg, criterion=self.loss_criterion, PE_Name=self.PE_Name)

        self.tb_writer_dict = {"writer": SummaryWriter(self.tb_save_folder),
                               "global_steps": 0}

        self.train()

    def train(self):
        logger = logging.getLogger(__name__)
        self.model_resume()
        val_maxmAP = 0
        if len(self.GPUS) > 1:
            self.model = torch.nn.DataParallel(self.model)
        for epoch in range(self.begin_epoch, self.end_epoch):
            sigma = adjust_sigma(epoch, self.cfg.MODEL.SIGMA, self.args.sigma_schedule)
            self.dataloader.dataset.sigma = sigma
            # train
            logger.info('=> Start train epoch {} lr {}'.format(epoch, self.optimizer.defaults['lr']))
            self.core_function.train(model=self.model, epoch=epoch, optimizer=self.optimizer,
                                     dataloader=self.dataloader,
                                     tb_writer_dict=self.tb_writer_dict, lr_scheduler=self.lr_scheduler)
            # save model
            if epoch % self.save_model_per_epoch == 0:
                model_save_path = self.save_model(epoch)
                logger.info('=> Saved epoch {} model state to {}'.format(epoch, model_save_path))
            if epoch % 1 == 0:
                val = Eval(phase='validate')
                mAP, model = val.eval()
                if val_maxmAP < mAP:
                    model_save_path = self.save_best_model(epoch, mAP)
                    val_maxmAP = mAP
                    logger.info(
                        '=> Saved best mAP {} epoch {} model state to {}'.format(val_maxmAP, epoch, model_save_path))

    def model_resume(self):
        logger = logging.getLogger(__name__)
        checkpoint_file = get_latest_checkpoint(self.checkpoints_save_folder)
        if checkpoint_file is not None:
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            self.model, self.optimizer, self.begin_epoch, ext_dict = resume(self.model, self.optimizer, checkpoint_file,
                                                                            gpus=self.GPUS)
            self.tb_writer_dict["global_steps"] = ext_dict["tensorboard_global_steps"]
        else:
            logger.warning("=> no checkpoint file available to resume")

    def save_model(self, epoch):
        model_save_path = save_checkpoint(epoch, self.checkpoints_save_folder, self.model, self.optimizer,
                                          global_steps=self.tb_writer_dict["global_steps"])
        return model_save_path

    def save_best_model(self, epoch, mAP):
        model_save_path = save_best_checkpoint(epoch, self.checkpoints_save_folder, self.model, self.optimizer, mAP,
                                               global_steps=self.tb_writer_dict["global_steps"])
        return model_save_path


if __name__ == '__main__':
    Train()
