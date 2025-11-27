import logging
from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class CNN1D(BaseModel):
    """Simple 1-D CNN baseline for hydrologic time series.

    This model:
      - uses the same InputLayer as the Transformer/LSTM
      - applies temporal 1-D convolutions over the embedded sequence
      - passes the result through the standard NH head to produce `y_hat`.

    Expected input from NH:
      data: Dict with keys like 'x_d', 'x_s', etc., handled by InputLayer.
    """

    # Parts that can be frozen/finetuned individually
    module_parts = ['embedding_net', 'cnn', 'head']

    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)

        # 1) Embedding net (same idea as Transformer)
        self.embedding_net = InputLayer(cfg)
        emb_size = self.embedding_net.output_size

        # 2) Hyperparameters for the CNN
        #    If not present in the config, we use defaults.
        hidden_size = getattr(cfg, "cnn_hidden_size", 64)
        kernel_size = getattr(cfg, "cnn_kernel_size", 5)
        n_layers    = getattr(cfg, "cnn_n_layers", 2)

        if kernel_size % 2 == 0:
            LOGGER.warning("cnn_kernel_size is even; consider using an odd kernel size for symmetric padding.")

        padding = kernel_size // 2  # simple 'same' padding

        layers = []
        in_channels = emb_size
        for i in range(n_layers):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=padding
            )
            layers.append(conv)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(getattr(cfg, "cnn_dropout", 0.0)))
            in_channels = hidden_size

        self.cnn = nn.Sequential(*layers)

        # 3) Head: maps [batch, seq, hidden_size] -> predictions (y_hat)
        #    self.output_size is set in BaseModel from target_variables
        self.head = get_head(cfg=cfg, n_in=hidden_size, n_out=self.output_size)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the CNN model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            NH input dictionary.

        Returns
        -------
        Dict[str, torch.Tensor]
            Must at least contain `y_hat` with shape [batch, seq, n_targets].
        """
        # 1) Embed dynamic + static inputs
        #    InputLayer in NH returns [seq_len, batch, emb_size]
        x = self.embedding_net(data)  # [seq, batch, emb]

        # 2) Prepare for Conv1d: [batch, emb, seq]
        x = x.permute(1, 2, 0)

        # 3) Temporal convolutions
        x = self.cnn(x)  # [batch, hidden_size, seq]

        # 4) Back to [batch, seq, hidden_size]
        x = x.permute(0, 2, 1)

        # 5) Use NH head to produce y_hat
        pred = self.head(x)  # dict with 'y_hat', etc.

        # (Optional) You can also return embeddings if you want:
        pred['embedding'] = x

        return pred
