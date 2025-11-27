import logging
from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class InformerEncoderLayer(nn.Module):
    """Simplified Informer-style encoder layer.

    Uses MultiheadAttention + FFN, with optional "distillation"
    (downsampling in time) after the layer, similar in spirit to
    Informer's sequence length reduction.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float,
                 distill: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # distillation (sequence length halving) via conv1d stride 2
        self.distill = distill
        if distill:
            self.conv_down = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                padding=1,
                stride=2
            )

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        # src: [seq, batch, d_model]
        attn_out, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout(attn_out))

        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout(ff))

        if self.distill:
            # [seq, batch, dim] -> [batch, dim, seq]
            x = src.permute(1, 2, 0)
            x = self.conv_down(x)  # [batch, dim, seq//2]
            src = x.permute(2, 0, 1)  # back to [seq_new, batch, dim]

        return src


class InformerEncoder(nn.Module):
    """Stack of InformerEncoderLayer with progressive distillation."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, nlayers: int, last_no_distill: bool = True):
        super().__init__()

        layers = []
        for i in range(nlayers):
            # last layer optionally without distillation to keep final seq length
            distill = not (last_no_distill and i == nlayers - 1)
            layers.append(
                InformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    distill=distill
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = src
        mask = src_mask
        for layer in self.layers:
            out = layer(out, mask)
            # After distillation, sequence length shrinks; mask (if used) should
            # be recomputed upstream if you need strict causal behaviour.
            # For now we don't change mask, which is a mild approximation.
        return out


class InformerModel(BaseModel):
    """Informer-style model integrated into NeuralHydrology.

    - Uses InputLayer to embed dynamic + static inputs.
    - Applies InformerEncoder (attention + distillation).
    - Uses standard NH head to produce `y_hat`.
    """

    module_parts = ['embedding_net', 'encoder', 'head']

    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)
        d_model = self.embedding_net.output_size

        nhead = cfg.transformer_nheads
        dim_ff = cfg.transformer_dim_feedforward
        dropout = cfg.transformer_dropout
        nlayers = cfg.transformer_nlayers

        self.encoder = InformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            nlayers=nlayers,
            last_no_distill=True
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=d_model, n_out=self.output_size)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # InputLayer returns [seq, batch, d_model]
        x = self.embedding_net(data)

        # For simplicity, we skip an explicit causal mask here (NH already uses
        # seq-to-one/seq-to-seq training windows). You could add one if needed.
        src_mask = None

        encoded = self.encoder(x, src_mask)   # [seq', batch, d_model]

        # NH head expects [batch, seq, features]
        encoded_bsf = encoded.permute(1, 0, 2)
        out = self.head(self.dropout(encoded_bsf))

        out['embedding'] = x
        out['encoded'] = encoded
        return out
