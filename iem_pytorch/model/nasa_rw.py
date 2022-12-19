from torch import nn
from torch.nn.parameter import Parameter
import torch as pt
import os
import sys
repo_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_dir)
from model.utils import StateEncoder, StateDecoder, MLPList


class NASATransAE(nn.Module):
    def __init__(self,
                 in_dim: int,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 enc_nlayers: int,

                 tg_in_dim: int,
                 tg_d_model: int,
                 tg_decod_hids: list,
                 out_dim: int,

                 tg_nhead: int,
                 tg_d_hid: int,
                 tg_enc_nlayers: int,

                 dropout: float = 0.5,
                 ):
        super(NASATransAE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.ec_tokenizer = nn.Linear(in_dim, d_model)
        self.encoder = StateEncoder(d_model=d_model,
                                    nhead=nhead,
                                    d_hid=d_hid,
                                    nlayers=enc_nlayers,
                                    dropout=dropout)
        self.bottleneck = nn.Linear(d_model, tg_d_model)

        self.de_tokenizer = nn.Linear(tg_in_dim, tg_d_model)
        self.decode = StateDecoder(d_model=tg_d_model,
                                   nhead=tg_nhead,
                                   d_hid=tg_d_hid,
                                   nlayers=tg_enc_nlayers,
                                   dropout=dropout)
        self.mlp = MLPList(tg_d_model,
                           tg_decod_hids,
                           out_dim)

        self.ins_norm = nn.InstanceNorm1d(tg_d_model)
        self.l_norm = nn.LayerNorm(tg_d_model)
        self.mean_token = Parameter(data=pt.ones((1, 1, d_model)),
                                    requires_grad=True)

    def gap(self, xs: pt.Tensor):
        [t, b, d] = xs.size()
        tokens = self.ec_tokenizer(xs)
        mean_token = self.mean_token.repeat(1, b, 1)
        tokens = pt.concat((tokens, mean_token), dim=0)
        feats = self.encoder(tokens)
        mean = feats[t:, :]
        # mean = self.bottleneck(mean)
        # mean = self.l_norm(mean)
        return mean

    def trans(self, xs):
        outs = tuple([x.permute(1, 0, 2) for x in xs])
        return outs

    def forward(self,
                src: pt.Tensor,
                sty: pt.Tensor,
                tg: pt.Tensor):
        """
        Args:
            src: [batch, time, feature_dims]
        Outs:
            outs: [batch, time, feature_dims]
        """
        [b, t, d] = src.size()
        if self.training:
            rand_idxs = pt.randperm(n=b)
            sty = sty[rand_idxs]

        # src
        src = src.permute(1, 0, 2)  # [b, t, d] -> [t, b, d]
        feats_mean = self.gap(src)

        # sty
        sty = sty.permute(1, 0, 2)  # [b, t, d] -> [t, b, d]
        sty_mean = self.gap(sty)

        # tf
        tg = tg.permute(1, 0, 2)
        tg_embs = self.de_tokenizer(tg)

        # recon
        sty_preds = self.mlp(self.decode(tg_embs, sty_mean))

        # src recon
        src_preds = self.mlp(self.decode(tg_embs, feats_mean))

        out_feats = feats_mean.repeat(t, 1, 1)

        outs = [out_feats,
                sty_mean, feats_mean,
                sty_preds, src_preds]
        return self.trans(outs)
