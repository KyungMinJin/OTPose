import torch.nn as nn
import torch


class ST_OHKW_MSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(ST_OHKW_MSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.mse_criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output_s, output_t, target, target_weight, effective_num_joints: int = None):
        batch_size = output_t.size(0)
        num_joints = output_t.size(1)
        heatmaps_pred_t = output_t.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_pred_s = output_s.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        if effective_num_joints is None:
            effective_num_joints = num_joints

        mse_loss = 0
        mse_loss_s = 0
        loss = []
        loss_s = []

        for idx in range(num_joints):
            heatmap_pred_t = heatmaps_pred_t[idx].squeeze()
            heatmap_pred_s = heatmaps_pred_s[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            if self.use_target_weight:
                # student
                if torch.max(heatmap_gt) == 1:
                    loss_s.append(0.5 * (self.criterion(
                        heatmap_pred_s.mul(target_weight[:, idx]),
                        heatmap_gt.mul(target_weight[:, idx])
                    )))
                    mse_loss_s += (self.mse_criterion(heatmap_pred_s.mul(target_weight[:, idx]),
                                                      heatmap_gt.mul(target_weight[:, idx])))
                else:
                    loss_s.append(0.5 * (self.criterion(
                        heatmap_pred_s.mul(target_weight[:, idx]),
                        heatmap_gt.mul(target_weight[:, idx])
                    ) + self.criterion(
                        heatmap_pred_s.mul(target_weight[:, idx]),
                        heatmap_pred_t.mul(target_weight[:, idx])
                    )))
                    mse_loss_s += (self.mse_criterion(heatmap_pred_s.mul(target_weight[:, idx]),
                                                      heatmap_gt.mul(target_weight[:, idx]))
                                   + self.mse_criterion(heatmap_pred_s.mul(target_weight[:, idx]),
                                                        heatmap_pred_t.mul(target_weight[:, idx])))

                # # teachher
                # loss.append(0.5 * self.criterion(
                #     heatmap_pred_t.mul(target_weight[:, idx]),
                #     heatmap_gt.mul(target_weight[:, idx])
                # ))
                # mse_loss += self.mse_criterion(heatmap_pred_t.mul(target_weight[:, idx]),
                #                                heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss.append(0.5 * self.criterion(heatmap_pred_t, heatmap_gt))
                mse_loss += self.mse_criterion(heatmap_pred_t, heatmap_gt)

        # loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        # loss = self.ohkm(torch.cat(loss, dim=1))

        loss_s = [l.mean(dim=1).unsqueeze(dim=1) for l in loss_s]
        loss_s = self.ohkm(torch.cat(loss_s, dim=1))
        final_loss = loss_s + mse_loss_s

        # return {"ohkm_loss_t": loss,
        #         "mse_loss_t": mse_loss / effective_num_joints,
        #         "ohkm_loss_s": loss_s,
        #         "mse_loss_s": mse_loss_s / effective_num_joints,
        #         'final_loss': final_loss}
        return {"ohkm_loss_s": loss_s,
                "mse_loss_s": mse_loss_s / effective_num_joints,
                'final_loss': final_loss}


class JointsMSE_OHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsMSE_OHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.mse_criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight, effective_num_joints: int = None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        if effective_num_joints is None:
            effective_num_joints = num_joints

        mse_loss = 0
        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
                mse_loss += self.mse_criterion(heatmap_pred.mul(target_weight[:, idx]),
                                               heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )
                mse_loss += self.mse_criterion(heatmap_pred, heatmap_gt)

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = self.ohkm(torch.cat(loss, dim=1))
        final_loss = loss + mse_loss

        return {"ohkm_loss": loss,
                "mse_loss": mse_loss / effective_num_joints,
                'final_loss': final_loss}


class JointMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        # self.criterion = nn.MSELoss(reduction='elementwise_mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight, effective_num_joints: int = None, margin=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        if effective_num_joints is None:
            effective_num_joints = num_joints
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            # if self.use_target_weight and margin is not None:
            #     spatial_cont = ((margin[0] + 1) * (margin[1] + 1) / (
            #                 margin[0] * margin[1] + 2 * (margin[0] + margin[1]) + 3))[:, None]
            #     loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]).mul(spatial_cont),
            #                            heatmap_gt.mul(target_weight[:, idx]).mul(spatial_cont))
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
                # loss += self.criterion(torch.Tensor([torch.argmax(heatmap_pred)//output.size(2), torch.argmax(heatmap_pred)%output.size(2)]), torch.Tensor([torch.argmax(heatmap_gt)//output.size(2), torch.argmax(heatmap_gt)%output.size(2)]))

            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / effective_num_joints


def build_loss(cfg, **kwargs):
    if cfg.LOSS.NAME == "ST_OHKW_MSELoss":
        return ST_OHKW_MSELoss(cfg.LOSS.USE_TARGET_WEIGHT)
    elif cfg.LOSS.NAME == "MSELOSS_OHKM":
        return JointsMSE_OHKMMSELoss(cfg.LOSS.USE_TARGET_WEIGHT)
