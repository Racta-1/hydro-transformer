import logging
from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class FourierBlock(nn.Module):
    """Simplified FEDformer-style Fourier mixing block.

    Applies FFT over the temporal axis, learns a linear transform in
    frequency space, and returns to time domain via inverse FFT.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.complex_weight = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq, batch, d_model]
        # FFT over time dimension (dim=0)
        x_f = torch.fft.rfft(x, dim=0)  # [freq, batch, d_model] complex

        # Linear mixing in feature dimension (per frequency)
        # x_f: [F, B, D], weight: [D, D]
        x_f = torch.einsum('fbd,dk->fbk', x_f, self.complex_weight.to(x_f.dtype))

        # Inverse FFT back to time domain
        x_time = torch.fft.irfft(x_f, n=x.size(0), dim=0)  # [seq, batch, d_model]
        return x_time


class FEDformerEncoderLayer(nn.Module):
    """Simplified FEDformer encoder layer using FourierBlock + FFN."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.fourier_block = FourierBlock(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Fourier mixing
        f_out = self.fourier_block(src)
        src = self.norm1(src + self.dropout(f_out))

        # Feed-forward
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout(ff))

        return src


class FEDformerEncoder(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float, nlayers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            FEDformerEncoderLayer(d_model, dim_feedforward, dropout)
            for _ in range(nlayers)
        ])

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        out = src
        for layer in self.layers:
            out = layer(out)
        return out


class FEDformerModel(BaseModel):
    """FEDformer-style model integrated with NeuralHydrology.

    - Uses InputLayer to embed static + dynamic inputs.
    - Applies stacked FEDformerEncoder layers (Fourier mixing).
    - Uses NH head to predict `y_hat`.
    """

    module_parts = ['embedding_net', 'encoder', 'head']

    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)
        d_model = self.embedding_net.output_size

        dim_ff = cfg.transformer_dim_feedforward
        dropout = cfg.transformer_dropout
        nlayers = cfg.transformer_nlayers

        self.encoder = FEDformerEncoder(
            d_model=d_model,
            dim_feedforward=dim_ff,
            dropout=dropout,
            nlayers=nlayers
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=d_model, n_out=self.output_size)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [seq, batch, d_model]
        x = self.embedding_net(data)

        encoded = self.encoder(x)  # [seq, batch, d_model]

        # Head expects [batch, seq, features]
        encoded_bsf = encoded.permute(1, 0, 2)
        out = self.head(self.dropout(encoded_bsf))

        out['embedding'] = x
        out['encoded'] = encoded
        return out
