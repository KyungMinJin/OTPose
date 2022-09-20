#!/usr/bin/python
# -*- coding:utf8 -*-
import time
import torch
import numpy as np
import os.path as osp
import logging
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter

from tabulate import tabulate

from utils.evaluate import accuracy, save_result_images, save_fusion_images, save_f_inv_images
from utils.bbox import cs2box
from utils.heatmap import get_final_preds, get_max_preds
from utils.images import tensor2im, draw_skeleton_in_origin_image
from utils.setup import create_folder
from utils.transform import reverse_transforms


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class CommonFunction:
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR

        if "criterion" in kwargs.keys():
            self.criterion = kwargs["criterion"]
        if "tb_log_dir" in kwargs.keys():
            self.tb_log_dir = kwargs["tb_log_dir"]
        if "writer_dict" in kwargs.keys():
            self.writer_dict = kwargs["writer_dict"]

        self.PE_Name = kwargs.get("PE_Name")
        ##
        self.max_iter_num = 0
        self.dataloader_iter = None
        self.tb_writer = None
        self.global_steps = 0
        self.DataSetName = str(self.cfg.DATASET.NAME).upper()

    def _print_name_value(self, name_value, full_arch_name):
        logger = logging.getLogger(__name__)
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)

        table_header = ["Model"]
        table_header.extend([name for name in names])
        table_data = [full_arch_name]
        table_data.extend(["{:.4f}".format(value) for value in values])

        table = tabulate([table_data], tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Result Table: \n" + table)

    def train(self, model, epoch, optimizer, dataloader, tb_writer_dict, clip_grad_l2norm=1.0, **kwargs):
        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]
        self.pe_w, self.pe_h = self.cfg.MODEL.HEATMAP_SIZE
        self.num_joints = self.cfg.MODEL.NUM_JOINTS

        lr_scheduler = kwargs["lr_scheduler"]
        logger = logging.getLogger(__name__)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        acc = AverageMeter()
        # switch to train mode
        model.train()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        end = time.time()

        for iter_step in range(self.max_iter_num):
            input_x, input_sup_A, input_sup_B, input_sup_AA, input_sup_BB, target_heatmaps, \
            target_heatmaps_weight, meta = next(self.dataloader_iter)  #
            # input_x, input_sup_A, input_sup_B, \
            # target_heatmaps, target_heatmaps_weight, meta = next(self.dataloader_iter)

            self._before_train_iter(input_x)

            data_time.update(time.time() - end)

            target_heatmaps = target_heatmaps.cuda(non_blocking=True)
            target_heatmaps_weight = target_heatmaps_weight.cuda(non_blocking=True)

            outputs = []
            if self.PE_Name == 'OTPOSE':
                margin_left = meta["margin_left"]
                margin_right = meta["margin_right"]
                margin_lleft = meta["margin_lleft"]
                margin_rright = meta["margin_rright"]
                margin = torch.stack([margin_left, margin_right, margin_lleft, margin_rright], dim=1).cuda()
                concat_input = torch.cat((input_x, input_sup_A, input_sup_B, input_sup_AA, input_sup_BB), 1).cuda()
                outputs = model(concat_input, margin=margin)
            elif self.PE_Name == 'POSETRANSFORMER':
                outputs = model(input_x.cuda())

            pred_heatmaps_t, pred_heatmaps_s = [], []
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                pred_heatmaps_t, _, _, _, _ = outputs[1].split(input_x.shape[0], dim=0)  # cur image
                pred_heatmaps_s = outputs[0]  # cur image
                loss = self.criterion(pred_heatmaps_s, pred_heatmaps_t, target_heatmaps, target_heatmaps_weight)
                occlusion = (target_heatmaps + outputs[2]) / 2
                context_encoding = outputs[4]
                loss['final_loss'] += self.criterion(context_encoding, context_encoding,
                                                     occlusion, target_heatmaps_weight)['final_loss']
            else:
                pred_heatmaps = outputs
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)

            # compute gradient and do update step
            optimizer.zero_grad()
            loss['final_loss'].backward()
            if clip_grad_l2norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_grad_l2norm
                )
            optimizer.step()
            lr_scheduler.step()
            # record learning_rate

            _, avg_acc, cnt, _ = accuracy(pred_heatmaps_s.detach().cpu().numpy(),
                                          target_heatmaps.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # For Tensorboard
            lr = lr_scheduler.get_last_lr()[0]
            global_step = epoch * self.max_iter_num + iter_step
            if self.tb_writer is not None:
                # learning rate (after stepping)
                self.tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )

                losses_tracker = {}
                tag_dict = {}
                # track all losses
                for key, value in loss.items():
                    # init meter if necessary
                    if key not in losses_tracker:
                        losses_tracker[key] = AverageMeter()
                    # update
                    losses_tracker[key].update(value.item())
                # all losses
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                self.tb_writer.add_scalars(
                    'train/all_losses',
                    tag_dict,
                    global_step
                )
                # final loss
                self.tb_writer.add_scalar(
                    'train/final_loss',
                    losses_tracker['final_loss'].val,
                    global_step
                )

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                epoch, iter_step, self.max_iter_num
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.5f} ({:.5f})'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4 += '\t{:s} {:.5f} ({:.5f})'.format(
                        key, value.val, value.avg
                    )
            self.tb_writer.add_scalar('train/acc', acc.val, self.global_steps)
            self.global_steps += 1
            tb_writer_dict["global_steps"] = self.global_steps

            if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= self.max_iter_num - 1:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                      'Sigma {sigma:.3f}\t'.format(epoch, iter_step, self.max_iter_num,
                                                   batch_time=batch_time,
                                                   speed=input_x.size(0) / batch_time.val,
                                                   data_time=data_time, loss=losses_tracker['final_loss'], acc=acc,
                                                   sigma=dataloader.dataset.sigma)

                logger.info(msg)
                logger.info('\t'.join([block1, block2, block3, block4]))

                B, C, H, W = target_heatmaps.shape
                ridx = np.random.randint(0, B)
                rough_heatmaps = outputs[1]
                current_heatmaps, prev_heatmaps, next_heatmaps, _, _ = rough_heatmaps.split(input_x.shape[0], dim=0)

                fusion_heatmaps = outputs[2].detach().cpu().numpy()  # mask
                gf_heatmaps = outputs[3].detach().cpu().numpy()  # acc^2
                prev_subtractions = outputs[4].detach().cpu().numpy()  # velocity
                squeezeds = outputs[5].detach().cpu().numpy()
                total_flows = outputs[6].detach().cpu().numpy()
                pred_heatmaps = pred_heatmaps_s.detach().cpu().numpy()
                gt_heatmaps = target_heatmaps.detach().cpu().numpy()
                current_heatmaps = current_heatmaps.detach().cpu().numpy()
                prev_heatmaps = prev_heatmaps.detach().cpu().numpy()
                next_heatmaps = next_heatmaps.detach().cpu().numpy()

                gt_heatmap = gt_heatmaps[ridx]
                pred_heatmap = pred_heatmaps[ridx]
                fusion_heatmap = fusion_heatmaps[ridx]
                gf_heatmap = gf_heatmaps[ridx]
                context_encoding = prev_subtractions[ridx]
                current_heatmap = current_heatmaps[ridx]
                prev_heatmap = prev_heatmaps[ridx]
                next_heatmap = next_heatmaps[ridx]
                squeezed = squeezeds[ridx]
                total_flow = total_flows[ridx]

                pose, pred_conf = get_max_preds(pred_heatmaps)
                cpose, cpred_conf = get_max_preds(current_heatmaps)
                ppose, ppred_conf = get_max_preds(prev_heatmaps)
                npose, npred_conf = get_max_preds(next_heatmaps)
                gt_pose, gt_conf = get_max_preds(gt_heatmaps)
                prev_s_pose, prev_s_conf = get_max_preds(prev_subtractions)
                squeezedpose, squeezed_conf = get_max_preds(squeezeds)
                tf_pose, tf_conf = get_max_preds(total_flows)

                img = np.transpose(input_x[ridx].detach().cpu().numpy(), (1, 2, 0))
                pimg = np.transpose(input_sup_A[ridx].detach().cpu().numpy(), (1, 2, 0))
                nimg = np.transpose(input_sup_B[ridx].detach().cpu().numpy(), (1, 2, 0))

                save_result_images(osp.join(self.output_dir, 'img'), img, squeezedpose[ridx], squeezed_conf[ridx],
                                   heatmaps=squeezed, name='squeezed_')
                save_result_images(osp.join(self.output_dir, 'img'), img, tf_pose[ridx], tf_conf[ridx],
                                   heatmaps=total_flow, name='total_flow_')
                save_result_images(osp.join(self.output_dir, 'img'), img, pose[ridx], pred_conf[ridx],
                                   heatmaps=pred_heatmap, name='pred_')
                save_result_images(osp.join(self.output_dir, 'img'), img, gt_pose[ridx], gt_conf[ridx],
                                   heatmaps=gt_heatmap, name='gt_')
                save_result_images(osp.join(self.output_dir, 'img', 'backbone'), img, cpose[ridx], cpred_conf[ridx],
                                   heatmaps=current_heatmap, name='h_c_')
                save_result_images(osp.join(self.output_dir, 'img', 'backbone'), pimg, ppose[ridx], ppred_conf[ridx],
                                   heatmaps=prev_heatmap, name='h_p_')
                save_result_images(osp.join(self.output_dir, 'img', 'backbone'), nimg, npose[ridx], npred_conf[ridx],
                                   heatmaps=next_heatmap, name='h_n_')
                save_fusion_images(osp.join(self.output_dir, 'img', 'fusion'), img,
                                   heatmaps=total_flow, name='total_flow_')
                save_fusion_images(osp.join(self.output_dir, 'img', 'fusion'), img,
                                   heatmaps=fusion_heatmap, name='intersection_')
                save_fusion_images(osp.join(self.output_dir, 'img', 'fusion'), img, heatmaps=gf_heatmap,
                                   name='acc_square_')
                save_fusion_images(osp.join(self.output_dir, 'img', 'fusion'), img, heatmaps=context_encoding,
                                   name='prev_velocity_')
                save_fusion_images(osp.join(self.output_dir, 'img', 'backbone'), img, heatmaps=current_heatmap,
                                   name='h_c_')
                save_fusion_images(osp.join(self.output_dir, 'img', 'backbone'), pimg, heatmaps=prev_heatmap,
                                   name='h_p_')
                save_fusion_images(osp.join(self.output_dir, 'img', 'backbone'), nimg, heatmaps=next_heatmap,
                                   name='h_n_')

    def eval(self, model, dataloader, tb_writer_dict, **kwargs):

        logger = logging.getLogger(__name__)

        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]

        batch_time, data_time, losses, acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        phase = kwargs.get("phase", 'val')
        epoch = kwargs.get("epoch", "specified_model")
        # switch to evaluate mode
        model.eval()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        dataset = dataloader.dataset
        # prepare data fro validate
        num_samples = len(dataset)
        all_preds = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=np.float)
        all_boxes = np.zeros((num_samples, 6))
        image_path = []
        filenames = []
        filenames_map = {}
        filenames_counter = 0
        imgnums = []
        idx = 0
        acc_threshold = 0.7

        ###
        result_output_dir, vis_output_dir = self.vis_setup(logger, phase, epoch)
        ###

        with torch.no_grad():
            end = time.time()
            num_batch = len(dataloader)
            for iter_step in range(self.max_iter_num):
                input_x, input_sup_A, input_sup_B, input_sup_AA, input_sup_BB, target_heatmaps, \
                target_heatmaps_weight, meta = next(self.dataloader_iter)
                # input_x, input_sup_A, input_sup_B, target_heatmaps, target_heatmaps_weight, meta = next(
                #     self.dataloader_iter)

                if phase == 'validate':
                    self._before_val_iter(input_x)

                data_time.update(time.time() - end)
                # prepare model input
                margin_left = meta["margin_left"]
                margin_right = meta["margin_right"]
                margin_lleft = meta["margin_lleft"]
                margin_rright = meta["margin_rright"]
                margin = torch.stack([margin_left, margin_right, margin_lleft, margin_rright], dim=1).cuda()
                concat_input = torch.cat((input_x, input_sup_A, input_sup_B, input_sup_AA, input_sup_BB), 1).cuda()

                # margin = torch.stack([margin_left, margin_right], dim=1).cuda()
                target_heatmaps = target_heatmaps.cuda(non_blocking=True)

                # concat_input = torch.cat((input_x, input_sup_A, input_sup_B), 1).cuda()
                outputs = model(concat_input, margin=margin)

                if phase == 'validate':
                    if isinstance(model, torch.nn.DataParallel):
                        vis_dict = getattr(model.module, "vis_dict", None)
                    else:
                        vis_dict = getattr(model, "vis_dict", None)

                    if vis_dict:
                        self._running_val_iter(vis_dict=vis_dict,
                                               model_input=[input_x, input_sup_A, input_sup_B])
                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    pred_heatmaps = outputs[0]  # cur image
                    pred_heatmaps_t, _, _, _, _ = outputs[1].split(input_x.shape[0], dim=0)  # cur image
                else:
                    pred_heatmaps = outputs

                _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(),
                                              target_heatmaps.detach().cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= (num_batch - 1):
                    msg = '{}: [{}/{}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(phase, iter_step, num_batch,
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, acc=acc)
                    logger.info(msg)
                    ridx = np.random.randint(0, input_x.shape[0])
                    pred_heatmaps_teach = pred_heatmaps_t.detach().cpu().numpy()
                    pred_heatmaps_v = pred_heatmaps.detach().cpu().numpy()
                    gt_heatmaps = target_heatmaps.detach().cpu().numpy()

                    gt_heatmap = gt_heatmaps[ridx]
                    pred_heatmap_v = pred_heatmaps_v[ridx]
                    pred_heatmap_teach = pred_heatmaps_teach[ridx]

                    pose, pred_conf = get_max_preds(pred_heatmaps_v)
                    pose_t, pred_conf_t = get_max_preds(pred_heatmaps_teach)
                    gt_pose, gt_conf = get_max_preds(gt_heatmaps)

                    img = np.transpose(input_x[ridx].detach().cpu().numpy(), (1, 2, 0))

                    save_result_images(vis_output_dir, img, pose_t[ridx], pred_conf_t[ridx],
                                       heatmaps=pred_heatmap_teach, name='teacher_pred_')
                    save_result_images(vis_output_dir, img, pose[ridx], pred_conf[ridx],
                                       heatmaps=pred_heatmap_v, name='pred_')
                    save_result_images(vis_output_dir, img, gt_pose[ridx], gt_conf[ridx],
                                       heatmaps=gt_heatmap, name='gt_')

                # for eval
                for ff in range(len(meta['image'])):
                    cur_nm = meta['image'][ff]
                    if not cur_nm in filenames_map:
                        filenames_map[cur_nm] = [filenames_counter]
                    else:
                        filenames_map[cur_nm].append(filenames_counter)
                    filenames_counter += 1

                center = meta['center'].numpy()
                scale = meta['scale'].numpy()
                score = meta['score'].numpy()
                num_images = input_x.size(0)

                preds, maxvals = get_final_preds(pred_heatmaps.detach().cpu().numpy(), center, scale)
                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])
                idx += num_images

                # tensorboard writ
                self.global_steps += 1
                #
                self._after_val_iter(meta["image"], preds, maxvals, vis_output_dir, center, scale)

        logger.info('########################################')
        logger.info('{}'.format(self.cfg.EXPERIMENT_NAME))

        name_values, perf_indicator = dataset.evaluate(self.cfg, all_preds, result_output_dir, all_boxes,
                                                       filenames_map, filenames, imgnums)

        model_name = self.cfg.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                self._print_name_value(name_value, model_name)
        else:
            self._print_name_value(name_values, model_name)

        tb_writer_dict["global_steps"] = self.global_steps
        return perf_indicator

    def _before_train_iter(self, batch_x):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "train_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _before_val_iter(self, batch_x):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "val_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _running_val_iter(self, **kwargs):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        vis_dict = kwargs.get("vis_dict")
        #
        show_image_num = min(3, len(vis_dict["current_x"]))
        current_x = vis_dict["current_x"][0:show_image_num]  # [N,3,384,288]
        previous_x = vis_dict["previous_x"][0:show_image_num]  # [N,3,384,288]
        next_x = vis_dict["next_x"][0:show_image_num]  # [N,3,384,288]
        current_rough_heatmaps = vis_dict["current_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        previous_rough_heatmaps = vis_dict["previous_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        next_rough_heatmaps = vis_dict["next_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        diff_A = vis_dict["diff_A"][0:show_image_num]  # [N,17,96,72]
        diff_B = vis_dict["diff_B"][0:show_image_num]  # [N,17,96,72]
        diff_heatmaps = vis_dict["diff_heatmaps"][0:show_image_num]  # [N,34,96,72]
        support_heatmaps = vis_dict["support_heatmaps"][0:show_image_num]  # [N,17,96,72]
        prf_ptm_combine_featuremaps = vis_dict["prf_ptm_combine_featuremaps"][0:show_image_num]  # [N,96,96,72]
        warped_heatmaps_list = [warped_heatmaps[0:show_image_num] for warped_heatmaps in
                                vis_dict["warped_heatmaps_list"]]  # [N,17,96,72]
        output_heatmaps = vis_dict["output_heatmaps"][0:show_image_num]  # [N,17,96,72]

        show_three_input_image = make_grid(reverse_transforms(torch.cat([previous_x, current_x, next_x], dim=0)),
                                           nrow=show_image_num)
        self.tb_writer.add_image('01_three_input_image', show_three_input_image, global_step=self.global_steps)

        # show 2.
        three_rough_heatmaps_channels = []
        current_rough_heatmap_channels = current_rough_heatmaps.split(1, dim=1)
        previous_rough_heatmap_channels = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmap_channels = next_rough_heatmaps.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            three_rough_heatmaps_channels.append(current_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(previous_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(next_rough_heatmap_channels[i])

        three_heatmaps_tensor = torch.clamp_min(torch.cat(three_rough_heatmaps_channels, dim=0), 0)
        three_heatmaps_image = make_grid(three_heatmaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('02_three_heatmaps_image', three_heatmaps_image, global_step=self.global_steps)

        # show 3.
        two_diff_channels = []
        diff_A_channels = diff_A.split(1, dim=1)
        diff_B_channels = diff_B.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            two_diff_channels.append(diff_A_channels[i])
            two_diff_channels.append(diff_B_channels[i])

        two_diff_channels_tensor = torch.clamp_min(torch.cat(two_diff_channels, dim=0), 0)
        two_diff_image = make_grid(two_diff_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('03_two_diff_image', two_diff_image, global_step=self.global_steps)

        # show4.
        diff_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(diff_heatmaps, 1, dim=1), dim=0), 0)
        diff_heatmaps_channels_image = make_grid(diff_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('04_diff_heatmaps_channels_image', diff_heatmaps_channels_image,
                                 global_step=self.global_steps)

        # show5.
        support_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(support_heatmaps, 1, dim=1), dim=0), 0)
        support_heatmaps_channels_image = make_grid(support_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('05_support_heatmaps_channels_image', support_heatmaps_channels_image,
                                 global_step=self.global_steps)

        # show6.
        prf_ptm_combine_featuremaps_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(prf_ptm_combine_featuremaps, 1, dim=1), dim=0),
            0)
        prf_ptm_combine_featuremaps_channels_image = make_grid(prf_ptm_combine_featuremaps_channels_tensor,
                                                               nrow=show_image_num)
        self.tb_writer.add_image('06_prf_ptm_combine_featuremaps_channels_image',
                                 prf_ptm_combine_featuremaps_channels_image, global_step=self.global_steps)

        # show7.
        warped_heatmaps_1_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[0], 1, dim=1), dim=0), 0)
        warped_heatmaps_1_channels_image = make_grid(warped_heatmaps_1_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('07_warped_heatmaps_1_channels_image', warped_heatmaps_1_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_2_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[1], 1, dim=1), dim=0), 0)
        warped_heatmaps_2_channels_image = make_grid(warped_heatmaps_2_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('08_warped_heatmaps_2_channels_image', warped_heatmaps_2_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_3_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[2], 1, dim=1), dim=0), 0)
        warped_heatmaps_3_channels_image = make_grid(warped_heatmaps_3_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('09_warped_heatmaps_3_channels_image', warped_heatmaps_3_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_4_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[3], 1, dim=1), dim=0), 0)
        warped_heatmaps_4_channels_image = make_grid(warped_heatmaps_4_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('10_warped_heatmaps_4_channels_image', warped_heatmaps_4_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_5_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[4], 1, dim=1), dim=0), 0)
        warped_heatmaps_5_channels_image = make_grid(warped_heatmaps_5_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('11_warped_heatmaps_5_channels_image', warped_heatmaps_5_channels_image,
                                 global_step=self.global_steps)

        # show8.
        output_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(output_heatmaps, 1, dim=1), dim=0), 0)
        output_heatmaps_channels_image = make_grid(output_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('12_output_heatmaps_channels_image', output_heatmaps_channels_image,
                                 global_step=self.global_steps)

    def _after_val_iter(self, image, preds_joints, preds_confidence, vis_output_dir, center, scale):
        cfg = self.cfg
        # prepare data
        coords = np.concatenate([preds_joints, preds_confidence], axis=-1)
        bboxes = []
        for index in range(len(center)):
            xyxy_bbox = cs2box(center[index], scale[index], pattern="xyxy")
            bboxes.append(xyxy_bbox)

        if cfg.DEBUG.VIS_SKELETON or cfg.DEBUG.VIS_BBOX:
            draw_skeleton_in_origin_image(image, coords, bboxes, vis_output_dir, vis_skeleton=cfg.DEBUG.VIS_SKELETON,
                                          vis_bbox=cfg.DEBUG.VIS_BBOX)

    def vis_setup(self, logger, phase, epoch):
        if phase == 'test':
            prefix_dir = "test"
        elif phase == 'train':
            prefix_dir = "train"
        elif phase == 'validate':
            prefix_dir = "validate"
        else:
            prefix_dir = "inference"

        if isinstance(epoch, int):
            epoch = "model_{}".format(str(epoch))

        output_dir_base = osp.join(self.output_dir, epoch, prefix_dir,
                                   "use_gt_box" if self.cfg.VAL.USE_GT_BBOX else "use_precomputed_box")
        vis_output_dir = osp.join(output_dir_base, "vis")
        result_output_dir = osp.join(output_dir_base, "prediction_result")
        create_folder(vis_output_dir)
        create_folder(result_output_dir)
        logger.info("=> Vis Output Dir : {}".format(vis_output_dir))
        logger.info("=> Result Output Dir : {}".format(result_output_dir))

        if phase == 'validate':
            tensorboard_log_dir = osp.join(self.output_dir, epoch, prefix_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        if self.cfg.DEBUG.VIS_SKELETON:
            logger.info("=> VIS_SKELETON")
        if self.cfg.DEBUG.VIS_BBOX:
            logger.info("=> VIS_BBOX")
        return result_output_dir, vis_output_dir
