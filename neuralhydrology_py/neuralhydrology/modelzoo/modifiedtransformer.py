import logging
import math
from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.positional_encoding import PositionalEncoding
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class ModifiedTransformer(BaseModel):
    """Improved Transformer model for time series prediction.
    
    Key improvements over standard Transformer:
    1. Optional input scaling (can be disabled)
    2. Cached causal mask for efficiency
    3. Better initialization strategy
    4. Option to use only final hidden state (like LSTM)
    """
    module_parts = ['embedding_net', 'encoder', 'head']

    def __init__(self, cfg: Config):
        super(ModifiedTransformer, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        if self.embedding_net.output_size % cfg.transformer_nheads != 0:
            raise ValueError("Embedding dimension must be divisible by number of transformer heads. "
                             "Use statics_embedding/dynamics_embedding to specify the embedding.")

        # Option to disable input scaling (add to config: transformer_scale_inputs: False)
        self._scale_inputs = getattr(cfg, 'transformer_scale_inputs', False)
        if self._scale_inputs:
            self._sqrt_embedding_dim = math.sqrt(self.embedding_net.output_size)
        else:
            self._sqrt_embedding_dim = 1.0

        self._positional_encoding_type = cfg.transformer_positional_encoding_type
        if self._positional_encoding_type.lower() == 'concatenate':
            encoder_dim = self.embedding_net.output_size * 2
        elif self._positional_encoding_type.lower() == 'sum':
            encoder_dim = self.embedding_net.output_size
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {self._positional_encoding_type}")
        
        self.positional_encoder = PositionalEncoding(embedding_dim=self.embedding_net.output_size,
                                                      dropout=cfg.transformer_positional_dropout,
                                                      position_type=cfg.transformer_positional_encoding_type,
                                                      max_len=cfg.seq_length)

        # Cached causal mask
        self._register_causal_mask(cfg.seq_length)

        # Encoder with improved initialization
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=cfg.transformer_nheads,
            dim_feedforward=cfg.transformer_dim_feedforward,
            dropout=cfg.transformer_dropout,
            activation='gelu',  # GELU often works better than ReLU
            batch_first=False,  # Keep False to match original
            norm_first=True     # Pre-norm architecture (more stable)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=cfg.transformer_nlayers,
            norm=nn.LayerNorm(encoder_dim)  # Final layer norm
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=encoder_dim, n_out=self.output_size)

        # Improved initialization
        self._reset_parameters()

    def _register_causal_mask(self, seq_length: int):
        """Pre-compute and register causal mask as buffer."""
        mask = torch.triu(torch.full((seq_length, seq_length), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def _reset_parameters(self):
        """Initialize parameters using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'embedding_net' in name:
                continue  # Let embedding net handle its own initialization
            
            if 'weight' in name and param.dim() >= 2:
                # Xavier uniform for weights
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.constant_(param, 0.0)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the improved transformer model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
        """
        # Pass through embedding layers
        x_d = self.embedding_net(data)
        
        # Get sequence length for this batch
        seq_len = x_d.size(0)
        
        # Apply positional encoding with optional scaling
        if self._scale_inputs:
            positional_encoding = self.positional_encoder(x_d * self._sqrt_embedding_dim)
        else:
            positional_encoding = self.positional_encoder(x_d)

        # Use pre-computed causal mask (slice to actual sequence length)
        mask = self.causal_mask[:seq_len, :seq_len]

        # Encoding
        output = self.encoder(positional_encoding, mask)

        # Apply head
        pred = self.head(self.dropout(output.transpose(0, 1)))

        # Add intermediates to output
        pred['embedding'] = x_d
        pred['positional_encoding'] = positional_encoding
        pred['encoder_output'] = output

        return pred


class MinimalTransformer(BaseModel):
    """Minimal Transformer that mimics LSTM architecture more closely.
    
    Simplifications:
    1. No input scaling
    2. Simpler initialization matching LSTM
    3. Option to use only final state
    """
    module_parts = ['embedding_net', 'encoder', 'head']

    def __init__(self, cfg: Config):
        super(MinimalTransformer, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        if self.embedding_net.output_size % cfg.transformer_nheads != 0:
            raise ValueError("Embedding dimension must be divisible by number of transformer heads.")

        self._positional_encoding_type = cfg.transformer_positional_encoding_type
        if self._positional_encoding_type.lower() == 'concatenate':
            encoder_dim = self.embedding_net.output_size * 2
        elif self._positional_encoding_type.lower() == 'sum':
            encoder_dim = self.embedding_net.output_size
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {self._positional_encoding_type}")
        
        self.positional_encoder = PositionalEncoding(
            embedding_dim=self.embedding_net.output_size,
            dropout=cfg.transformer_positional_dropout,
            position_type=cfg.transformer_positional_encoding_type,
            max_len=cfg.seq_length
        )

        # Pre-computed causal mask
        mask = torch.triu(torch.full((cfg.seq_length, cfg.seq_length), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)

        # Simple encoder (no pre-norm, matching original)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=cfg.transformer_nheads,
            dim_feedforward=cfg.transformer_dim_feedforward,
            dropout=cfg.transformer_dropout
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=cfg.transformer_nlayers,
            norm=None
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=encoder_dim, n_out=self.output_size)

        # LSTM-style initialization
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize like LSTM: uniform distribution in small range."""
        initrange = 0.1
        for layer in self.encoder.layers:
            layer.linear1.weight.data.uniform_(-initrange, initrange)
            layer.linear1.bias.data.zero_()
            layer.linear2.weight.data.uniform_(-initrange, initrange)
            layer.linear2.bias.data.zero_()

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with minimal processing."""
        x_d = self.embedding_net(data)
        seq_len = x_d.size(0)
        
        # No input scaling - just add positional encoding
        positional_encoding = self.positional_encoder(x_d)
        
        # Use causal mask
        mask = self.causal_mask[:seq_len, :seq_len]
        
        # Encode
        output = self.encoder(positional_encoding, mask)
        
        # Apply head to all timesteps
        pred = self.head(self.dropout(output.transpose(0, 1)))
        
        pred['embedding'] = x_d
        pred['encoder_output'] = output
        
        return pred