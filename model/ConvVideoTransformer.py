from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F

from model.blocks import LayerNorm, get_sinusoid_encoding, TransformerBlock

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class ConvTransformer(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
            self,
            n_in,  # input feature dimension
            n_embd,  # embedding dimension (after convolution)
            n_head,  # number of head for self-attention in transformers
            n_embd_ks,  # conv kernel size of the embedding network
            max_len,  # max sequence length
            arch,  # (#convs, #stem transformers, #branch transformers)
            mha_win_size=[-1] * 6,  # size of local window for mha
            h=72,
            scale_factor=2,  # dowsampling rate for the branch,
            with_ln=True,  # if to attach layernorm after conv
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for the projection / MLP
            path_pdrop=0.0,  # droput rate for drop path
            use_abs_pe=True,  # use absolute position embedding
            use_rel_pe=False,  # use relative position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        # assert len(mha_win_size) == (1 + arch[2])
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        ###
        self.fpn_strides = [scale_factor ** i for i in range(arch[-1] + 1)]
        if isinstance(mha_win_size, int):
            self.mha_win_size = [mha_win_size] * len(self.fpn_strides)

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
            # self.register_buffer("pos_embd", pos_embd, persistent=False)
            self.register_buffer("pos_embd", pos_embd)

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(
                nn.Conv2d(in_channels, n_embd, n_embd_ks,
                          stride=1, padding=n_embd_ks // 2,
                          dilation=1, groups=1, bias=(not with_ln), padding_mode='zeros')
            )
            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(1, 1),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[0],
                use_rel_pe=self.use_rel_pe
            )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(TransformerBlock(
                n_embd, n_head,
                n_ds_strides=(self.scale_factor, self.scale_factor),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=self.mha_win_size[1 + idx],
                use_rel_pe=self.use_rel_pe
            )
            )
            self.upsample.append(nn.Upsample(scale_factor=2 ** (idx + 1), mode='linear'))

        self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) (h w)', p1=1, p2=1)
        self.inv_patch_embedding = Rearrange('b (p1 p2 c) (h w) -> b c (h p1) (w p2)', p1=1, p2=1, h=h)

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    # def forward(self, x, mask):
    def forward(self, x):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, H, W = x.size()
        T = H * W

        # embedding network
        for idx in range(len(self.embd)):
            x = self.embd[idx](x)
            # x, mask = self.embd[idx](x, mask)
            x = self.to_patch_embedding(x)
            x = self.relu(self.embd_norm[idx](x))
            x = self.inv_patch_embedding(x)

        x = self.to_patch_embedding(x)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T]
            # x = x + pe[:, :, :T] * mask.float()

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T]
            # x = x + pe[:, :, :T] * mask.float()

        # stem transformer
        for idx in range(len(self.stem)):
            x = self.stem[idx](x)
            # x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = tuple()
        # out_masks = tuple()
        # 1x resolution
        out_feats += (x, )
        # out_feats += (self.inv_patch_embedding(x),)
        # out_masks += (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x = self.branch[idx](x)
            # x, mask = self.branch[idx](x, mask)
            # out_feats += (x, )

            # rearrange = Rearrange('b (p1 p2 c) (h w) -> b c (h p1) (w p2)', p1=1, p2=1, h=H // 2 ** (idx + 1))
            # out_feats += (self.upsample[idx](rearrange(x)),)
            out_feats += (self.upsample[idx](x),)
            # print(self.upsample[idx](rearrange(x)).shape)
            # out_masks += (mask, )

        # return out_feats[0]
        return out_feats
        # return out_feats, out_masks
