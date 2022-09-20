from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from model.ConvVideoTransformer import ConvTransformer
from model.RSB import CHAIN_RSB_BLOCKS
from model.layers import DeformableCONV
from thirdparty.deform_conv import DeformConv, ModulatedDeformConv

import copy
from typing import Optional
from model.HRNet import BasicBlock, Bottleneck, HRNet
from utils.heatmap import normalize_0_to_1

BN_MOMENTUM = 0.1


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos,
                                        src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask, pos=pos,
                               src_key_padding_mask=src_key_padding_mask)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", normalize_before=False, return_atten_map=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)  # (96+20)*2 => 192
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_atten_map = return_atten_map

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # src + pos
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # src = x + src2
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


def _mask_conv(nc, kh, kw, dd, dg):
    conv = nn.Conv2d(nc, dg * 1 * kh * kw, kernel_size=(3, 3), stride=(1, 1), dilation=(dd, dd),
                     padding=(1 * dd, 1 * dd), bias=False)
    return conv


def _offset_conv(nc, kh, kw, dd, dg):
    conv = nn.Conv2d(nc, dg * 2 * kh * kw, kernel_size=(3, 3), stride=(1, 1), dilation=(dd, dd),
                     padding=(1 * dd, 1 * dd), bias=False)
    return conv


class OTPose(nn.Module):
    def __init__(self, cfg, **kwargs):
        self.logger = logging.getLogger(__name__)

        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(OTPose, self).__init__()

        self.num_frames = 8
        self.d_seg = 0
        n_head = 1
        self.pe_w, self.pe_h = cfg.MODEL.HEATMAP_SIZE
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.patch_size = 1
        self.num_patches = (self.pe_h // self.patch_size) * (self.pe_w // self.patch_size)
        self.patch_dim = self.num_joints * self.patch_size ** 2
        self.temporal_encoding_dim = self.patch_dim * self.num_frames

        self.freeze_hrnet_weights = cfg.MODEL.FREEZE_HRNET_WEIGHTS
        self.pretrained = cfg.MODEL.PRETRAINED
        self.rough_pose_estimation_net = HRNet(cfg, phase="train" if self.training else "validate")

        # 6개 layer
        self.scale_arch = (0, 6, 2)
        self.flow_scale_arch = (0, 6, 0)
        self.max_seq_len = self.num_patches
        self.temporal_encoder1 = ConvTransformer(self.temporal_encoding_dim, self.temporal_encoding_dim,
                                                 n_head=2, n_embd_ks=3,
                                                 max_len=self.num_patches, arch=self.scale_arch,
                                                 proj_pdrop=0.1, path_pdrop=0.1, h=self.pe_h)
        self.temporal_encoder2 = ConvTransformer(self.temporal_encoding_dim, self.temporal_encoding_dim,
                                                 n_head=2, n_embd_ks=3,
                                                 max_len=self.num_patches, arch=self.scale_arch,
                                                 proj_pdrop=0.1, path_pdrop=0.1, h=self.pe_h)
        self.flow_encoder = ConvTransformer(self.patch_dim, self.patch_dim, n_head, 3, self.num_patches,
                                            arch=self.flow_scale_arch,
                                            proj_pdrop=0.1, path_pdrop=0.1, h=self.pe_h)

        self.deformable_conv_dilations = cfg.MODEL.DEFORMABLE_CONV.DILATION
        self.deformable_aggregation_type = cfg.MODEL.DEFORMABLE_CONV.AGGREGATION_TYPE

        self.final_layer1 = nn.Conv2d(
            in_channels=self.temporal_encoding_dim * (self.scale_arch[-1] + 1),
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)

        self.final_layer2 = nn.Conv2d(
            in_channels=self.temporal_encoding_dim * (self.scale_arch[-1] + 1),
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

        k = 3
        def_ch = cfg.MODEL.DEFORMABLE_CONV_CH
        offset_mask_combine_ch = cfg.MODEL.OFFSET_MASK_COMBINE_CONV
        self.offset_mask_combine_conv = CHAIN_RSB_BLOCKS(self.num_joints * 3, def_ch, offset_mask_combine_ch)
        self.def_fuse = CHAIN_RSB_BLOCKS(self.num_joints, self.num_joints, offset_mask_combine_ch)

        self.offsets_list, self.masks_list, self.modulated_deform_conv_list = [], [], []
        for d_index, dilation in enumerate(self.deformable_conv_dilations):
            # offsets
            offset_layers, mask_layers = [], []
            offset_layers.append(_offset_conv(def_ch, k, k, dilation, self.num_joints).cuda())
            mask_layers.append(_mask_conv(def_ch, k, k, dilation, self.num_joints).cuda())
            self.offsets_list.append(nn.Sequential(*offset_layers))
            self.masks_list.append(nn.Sequential(*mask_layers))
            self.modulated_deform_conv_list.append(DeformableCONV(self.num_joints, k, dilation))

        self.offsets_list = nn.ModuleList(self.offsets_list)
        self.masks_list = nn.ModuleList(self.masks_list)
        self.modulated_deform_conv_list = nn.ModuleList(self.modulated_deform_conv_list)

        self.init_weights()

    def _make_position_embedding(self, pe_type):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.spatial_pos_embedding = None
            self.temporal_pos_embedding = None
            self.logger.info("==> Without any PositionEmbedding~")
        else:
            if pe_type == 'learnable':
                self.flow_pos_embedding = nn.Parameter(
                    torch.randn(1, self.num_patches, self.patch_dim))
                self.temporal_pos_embedding = nn.Parameter(
                    torch.randn(1, self.num_patches, self.temporal_encoding_dim))
                self.logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.flow_pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(self.patch_dim),
                    requires_grad=False)
                self.temporal_pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(self.temporal_encoding_dim),
                    requires_grad=False)
                self.logger.info("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000, scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)

        return pos  # [1, h*w, d_model]

    def forward(self, x, **kwargs):
        num_color_channels = 3
        supplement = 5
        assert "margin" in kwargs
        margin = kwargs["margin"]

        if not x.is_cuda or not margin.is_cuda:
            x.cuda()
            margin.cuda()

        x = torch.cat(x.split(num_color_channels, dim=1), 0)
        true_batch_size = int(x.shape[0] / supplement)
        rough_heatmaps = self.rough_pose_estimation_net(x)
        current_heatmaps, prev_heatmaps, next_heatmaps, pprev_heatmaps, nnext_heatmaps = \
            rough_heatmaps.split(true_batch_size, dim=0)

        # 1
        total_b = current_heatmaps + prev_heatmaps + next_heatmaps + pprev_heatmaps + nnext_heatmaps
        squeezed = torch.sum(total_b, axis=1)
        _, c, _, _ = current_heatmaps.shape
        # squeezed = torch.stack([squeezed for _ in range(c)], dim=1) / supplement
        squeezed = torch.stack([squeezed for _ in range(c)], dim=1)
        # 2
        intersection = total_b * squeezed
        context_encoding = self.flow_encoder(total_b)
        context_encoding = torch.stack([s for s in context_encoding],
                        dim=1).contiguous().view(true_batch_size,
                                                 self.patch_dim * (self.flow_scale_arch[-1] + 1),
                                                 self.pe_h, self.pe_w)
        # context_encoding = context_encoding.contiguous().view(true_batch_size, self.patch_dim, self.pe_h, self.pe_w)

        # penalize
        prev_heatmaps = torch.div(prev_heatmaps, (margin.T[0] + 1)[:, None, None, None])
        next_heatmaps = torch.div(next_heatmaps, (margin.T[1] + 1)[:, None, None, None])
        pprev_heatmaps = torch.div(pprev_heatmaps, (margin.T[2] + 1)[:, None, None, None])
        nnext_heatmaps = torch.div(nnext_heatmaps, (margin.T[3] + 1)[:, None, None, None])

        # 3, 4, 5
        prev_b = current_heatmaps + (prev_heatmaps + pprev_heatmaps)
        next_b = current_heatmaps + (next_heatmaps + nnext_heatmaps)
        # 7, 8
        close_b = current_heatmaps + (next_heatmaps + prev_heatmaps)
        far_b = current_heatmaps + (nnext_heatmaps + pprev_heatmaps)

        prev_int = prev_b * squeezed
        next_int = next_b * squeezed
        close_int = close_b * squeezed
        far_int = far_b * squeezed

        x1 = torch.stack((intersection, context_encoding, prev_b, far_b, close_b,
                          prev_int, far_int, close_int), dim=2).flatten(start_dim=1, end_dim=2)
        x2 = torch.stack((intersection, context_encoding, next_b, close_b, far_b,
                          next_int, close_int, far_int), dim=2).flatten(start_dim=1, end_dim=2)
        x1 = self.temporal_encoder1(x1)
        x2 = self.temporal_encoder2(x2)
        x1 = torch.stack([s for s in x1],
                        dim=1).contiguous().view(true_batch_size,
                                                 self.temporal_encoding_dim * (self.scale_arch[-1] + 1),
                                                 self.pe_h, self.pe_w)
        x2 = torch.stack([s for s in x2],
                        dim=1).contiguous().view(true_batch_size,
                                                 self.temporal_encoding_dim * (self.scale_arch[-1] + 1),
                                                 self.pe_h, self.pe_w)
        # x1 = x1.contiguous().view(true_batch_size, self.temporal_encoding_dim, self.pe_h, self.pe_w)
        # x2 = x2.contiguous().view(true_batch_size, self.temporal_encoding_dim, self.pe_h, self.pe_w)
        x1 = self.final_layer1(x1)
        x2 = self.final_layer2(x2)

        branches = torch.cat([x1, x2], dim=1)
        def_heatmaps = self.def_fuse(total_b).cuda()

        trans_heatmaps = self.offset_mask_combine_conv(torch.cat([branches, def_heatmaps], dim=1))

        warped_heatmaps_list = []
        for d_index, dilation in enumerate(self.deformable_conv_dilations):
            offsets = self.offsets_list[d_index](trans_heatmaps)
            masks = self.masks_list[d_index](trans_heatmaps)
            warped_heatmaps = self.modulated_deform_conv_list[d_index](def_heatmaps, offsets, masks)
            warped_heatmaps_list.append(warped_heatmaps)

        if self.deformable_aggregation_type == "weighted_sum":

            warper_weight = 1 / len(self.deformable_conv_dilations)
            output_heatmaps = warper_weight * warped_heatmaps_list[0]
            for warper_heatmaps in warped_heatmaps_list[1:]:
                output_heatmaps += warper_weight * warper_heatmaps

        return output_heatmaps, rough_heatmaps, intersection, prev_b, context_encoding, squeezed, total_b

    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        return batched_inputs, torch.stack([batched_masks for _ in range(feats[0].shape[0])], dim=1)

    def init_weights(self):
        logger = logging.getLogger(__name__)
        ## init_weights
        rough_pose_estimation_name_set = set()
        for module_name, module in self.named_modules():
            # rough_pose_estimation_net
            if module_name.split('.')[0] == "rough_pose_estimation_net":
                rough_pose_estimation_name_set.add(module_name)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)

                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, DeformConv):
                filler = torch.zeros(
                    [module.weight.size(0), module.weight.size(1), module.weight.size(2), module.weight.size(3)],
                    dtype=torch.float32, device=module.weight.device)
                for k in range(module.weight.size(0)):
                    filler[k, k, int(module.weight.size(2) / 2), int(module.weight.size(3) / 2)] = 1.0
                module.weight = torch.nn.Parameter(filler)
                # module.weight.requires_grad = True
            elif isinstance(module, ModulatedDeformConv):
                filler = torch.zeros(
                    [module.weight.size(0), module.weight.size(1), module.weight.size(2), module.weight.size(3)],
                    dtype=torch.float32, device=module.weight.device)
                for k in range(module.weight.size(0)):
                    filler[k, k, int(module.weight.size(2) / 2), int(module.weight.size(3) / 2)] = 1.0
                module.weight = torch.nn.Parameter(filler)
                # module.weight.requires_grad = True
            else:
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
                    if name in ['weights']:
                        nn.init.normal_(module.weight, std=0.001)

        if os.path.isfile(self.pretrained):
            pretrained_state_dict = torch.load(self.pretrained)
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']
            logger.info('=> loading pretrained model {}'.format(self.pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] == '*':
                    layer_name = name.split('.')[0]
                    if layer_name in rough_pose_estimation_name_set:
                        need_init_state_dict[name] = m
                    else:
                        new_layer_name = "rough_pose_estimation_net.{}".format(layer_name)
                        if new_layer_name in rough_pose_estimation_name_set:
                            parameter_name = "rough_pose_estimation_net.{}".format(name)
                            need_init_state_dict[parameter_name] = m
            # TODO pretrained from posewarper not test
            self.load_state_dict(need_init_state_dict, strict=False)
        elif self.pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(self.pretrained))

        # rough_pose_estimation
        if self.freeze_hrnet_weights:
            self.rough_pose_estimation_net.freeze_weight()