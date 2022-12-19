from torch import nn
import torch.nn.functional as F
import torch as pt
import math


class SliceByCols(nn.Module):
    def __init__(self,
                 dim: int,
                 keep_idxs: list):
        super(SliceByCols, self).__init__()
        self.dim = dim
        self.keep_idxs = keep_idxs

    def forward(self, xs: pt.Tensor):
        # outs = []
        s_xs = pt.split(xs, 1, self.dim)
        outs = [s_xs[idx] for idx in self.keep_idxs]
        return tuple(outs)


class ResNormLayer(nn.Module):
    def __init__(self, linear_size,):
        super(ResNormLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.GELU()
        self.nonlin2 = nn.GELU()
        self.norm_fn1 = nn.LayerNorm(self.l_size)
        self.norm_fn2 = nn.LayerNorm(self.l_size)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.norm_fn1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        y = self.norm_fn2(y)
        out = x + y
        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MLPList(nn.Module):
    def __init__(self,
                 in_dim: int,
                 d_hids: list,
                 out_dim: int,
                 ):
        super(MLPList, self).__init__()
        self.mlps = []
        self.mlps.extend([nn.Linear(in_dim, d_hids[0]), nn.GELU()])
        for idx, d_hid in enumerate(d_hids):
            if (idx + 1) == len(d_hids):
                break
            self.mlps.append(nn.Linear(d_hid, d_hids[idx + 1]))
            self.mlps.append(nn.GELU())
        self.mlps.append(nn.Linear(d_hids[-1], out_dim))
        self.mlps = nn.ModuleList(self.mlps)

    def forward(self, src: pt.Tensor):
        """
        Args:
            src: [batch, time, feature_dims]
        Outs:
            outs: [batch, time, feature_dims]
        """
        for i, l in enumerate(self.mlps):
            src = l(src)
        return src


class ResMLP(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, input_dim, hidden_dim=None,
                 output_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_dim, input_dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Tokenizer(nn.Module):
    def __init__(self, in_dim, d_model):
        super(Tokenizer, self).__init__()
        self.d_model = d_model
        self.conv = nn.Conv1d(in_dim,
                              in_dim,
                              3,
                              padding='same',
                              )
        self.linear = nn.Conv1d(in_dim,
                                d_model,
                                1,
                                padding='same',
                                )

    def forward(self, x):
        # [q, b, d] -> [b, d, q]
        #x = x.permute(1, 2, 0)
        # [b, q, d] -> [b, d, q]
        x = x.permute(0, 2, 1)
        x = x + self.conv(x)
        x = self.linear(x)
        # [b, d, q] -> [q, b, d]
        # x = x.permute(2, 0, 1)
        # [b, d, q] -> [b, q, d]
        x = x.permute(0, 2, 1)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = pt.arange(max_len).unsqueeze(1)
        div_term = pt.exp(pt.arange(0, d_model, 2)
                          * (-math.log(10000.0) / d_model))
        pe = pt.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = pt.sin(position * div_term)
        pe[:, 0, 1::2] = pt.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: pt.Tensor, using_mask: pt.Tensor) -> pt.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)] * using_mask
        return x


class StateEncoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 dropout: float = 0.5):
        super().__init__()
        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model,
                                                    nhead,
                                                    d_hid,
                                                    dropout,
                                                    activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layers,
                                             nlayers)

    def forward(self, src, src_mask=None):
        feature = self.encoder(src, src_mask)
        return feature


class StateDecoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 dropout: float = 0.5):
        super().__init__()
        # Encoder
        decoder_layers = nn.TransformerDecoderLayer(d_model,
                                                    nhead,
                                                    d_hid,
                                                    dropout,
                                                    activation='gelu')
        self.decoder = nn.TransformerDecoder(decoder_layers,
                                             nlayers)

    def forward(self, src, memory, src_mask=None, memory_mask=None):
        feature = self.decoder(src, memory, src_mask, memory_mask)
        return feature
