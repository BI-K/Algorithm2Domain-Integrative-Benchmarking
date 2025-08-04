###################################################################################################
###################################################################################################
# From https://github.com/nhinrichsberlin/icu-vital-parameter-forecasting                        ##
# Published under MIT license.                                                                   ## 
# Hinrichs N, Roeschl T, Lanmueller P, Balzer F, Eickhoff C, O’Brien B, et al.                   ##
# Short-term vital parameter forecasting in the intensive care unit: A benchmark study           ##
# leveraging data from patients after cardiothoracic surgery. Barage S, editor.                  ##
# PLOS Digit Health. 2024 Sep 12;3(9):e0000598.                                                  ##
###################################################################################################
# based on https://github.com/gzerveas/mvts_transformer                                          ##
# with minor adjustments to fit the forecast project                                             ##
###################################################################################################



from torch import nn, zeros
from typing import Optional
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F    # noqa
from torch.nn.modules import (
    MultiheadAttention,
    Linear,
    Dropout,
    BatchNorm1d,
    TransformerEncoderLayer
)

class GRUHinrichs(nn.Module):
    def __init__(
            self, configs):

        super().__init__()

        # number of time-series in the input
        self.n_input_ts = configs.input_channels

        # number of neurons in the hidden layer
        self.hidden_size = configs.hidden_size_gru

        # number of layers in the network
        self.num_layers = configs.num_layers

        # dropout (only used in case of multiple rnn-layers)
        self.dropout = configs.dropout if self.num_layers > 1 else 0

        # number of time-series to predict
        self.n_output_ts = configs.num_cont_output_channels

        # number of future steps to predict
        # TODO make felxible in the future
        self.horizon = 1

        # RNN-layers
        self.gru = nn.GRU(input_size=self.n_input_ts,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout,
                          batch_first=True)

        # a linear output layer
        self.output_layer = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.n_output_ts * self.horizon
        )

    def forward(self, x, padding_masks=None):   # noqa. Accept parameter padding masks for compatibility

        # Initializing hidden state for first input with zeros
        h0 = zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)

        # shape of x: (batch_size, seq_len, n_input_ts)
        # shape of gru_out: (batch_size, seq_len, hidden_size)
        gru_out, _ = self.gru(x, h0.detach())

        # shape of output: (batch_size, seq_len, horizon * n_output_ts)
        output = self.output_layer(gru_out)

        # re-shape to shape (batch_size, seq_len, horizon, n_output_ts)
        output = output.reshape(output.size(0), output.size(1), self.horizon, self.n_output_ts)

        return output


######################## Transformer #####################################
# TODO integrate

# based on https://github.com/gzerveas/mvts_transformer
# with minor adjustments to fit the forecast project

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError(f"activation should be relu/gelu, not {activation}")


def _get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding
    raise NotImplementedError(f"pos_encoding should be 'learnable'/'fixed', not {pos_encoding}")


def _get_encoder_layer(norm, d_model, n_heads, dim_feedforward, dropout, activation):
    if norm == "LayerNorm":
        return TransformerEncoderLayer(d_model=d_model,
                                       nhead=n_heads,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       activation=activation)
    elif norm == "BatchNorm":
        return TransformerBatchNormEncoderLayer(d_model=d_model,
                                                nhead=n_heads,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation)
    raise ValueError(f"norm should be LayerNorm/BatchNorm, not {norm}")


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx}
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self,
                 d_model,
                 dropout=0.1,
                 max_len=1024,
                 scale_factor=1.0):

        super(FixedPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self,
                 d_model,
                 dropout=0.1,
                 max_len=1024):

        super(LearnablePositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multi-head attention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):

        super(TransformerBatchNormEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                **kwargs) -> Tensor:  # Accept any additional keyword arguments
    
        # Extract is_causal if present, default to False
        is_causal = kwargs.get('is_causal', False)
    
        # Check if MultiheadAttention supports is_causal parameter
        try:
            src2 = self.self_attn(src, src, src, 
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,
                              is_causal=is_causal)[0]
        except TypeError:
            # Fallback for older PyTorch versions
            src2 = self.self_attn(src, src, src, 
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
    
        src = src + self.dropout1(src2)
        src = src.permute(1, 2, 0)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = src.permute(1, 2, 0)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)
        return src


# Entry Point
class TransformerHinrichs(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, config):

        super(TransformerHinrichs, self).__init__()

        torch.manual_seed(123)
        # torch.use_deterministic_algorithms(True)

        self.max_len = config.sequence_len
        self.horizon = 1
        self.d_model = config.final_out_channels
        self.n_heads = config.n_heads
        self.num_layers = config.num_layers

        self.feat_dim = config.input_channels
        self.activation = config.activation
        self.dropout = config.dropout
        self.dim_feedforward = config.dim_feedforward

        self.project_inp = nn.Linear(self.feat_dim, self.d_model)

        pos_encoder = _get_pos_encoder(config.pos_encoding)
        self.pos_enc = pos_encoder(d_model=self.d_model,
                                   dropout=self.dropout * (1.0 - config.freeze),
                                   max_len=self.max_len)

        encoder_layer = _get_encoder_layer(norm=config.norm,
                                           d_model=self.d_model,
                                           n_heads=self.n_heads,
                                           dim_feedforward=self.dim_feedforward,
                                           dropout=self.dropout * (1.0 - config.freeze),
                                           activation=self.activation)
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False  # Add this line
        )

        self.act = _get_activation_fn(self.activation)
        self.dropout1 = nn.Dropout(self.dropout)

        self.feat_dim = self.feat_dim
        self.num_classes = config.final_out_channels
        #self.output_layer = nn.Linear(self.d_model, self.num_classes * self.horizon)
        self.output_layer = nn.Linear(self.max_len * self.d_model, config.features_len * self.num_classes)
       

    def forward(self, X, padding_masks=None):
        """
        Args:
            X: (batch_size, feat_dim, seq_length) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor,
             1 means keep vector at this position, 0 means padding
        Returns:
            output: (config.features_len * self.num_classes, batch_size)
        """

        if padding_masks is None:
            padding_masks = torch.ones(X.shape[0], X.shape[2], dtype=bool).to(X.device)

        # mask future steps - this creates a float tensor
        future_mask = nn.Transformer.generate_square_subsequent_mask(X.size(2)).to(X.device)

        # current shape [batch_size, feat_dim, seq_length] => [seq_len, batch_size, feat_dim]
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim].
        inp = X.permute(2,0,1)

        # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.project_inp(inp) * math.sqrt(self.d_model)

        # add positional encoding
        inp = self.pos_enc(inp)

        # Convert boolean padding mask to float to match future_mask type
        # NOTE: logic for padding masks is reversed to comply with definition
        # in MultiHeadAttention, TransformerEncoderLayer
        src_key_padding_mask = (~padding_masks).float()
        
        # Replace True (should be masked) with -inf and False (keep) with 0.0
        src_key_padding_mask = src_key_padding_mask.masked_fill(src_key_padding_mask == 1, float('-inf'))

        # (seq_length, batch_size, d_model)
        output = self.transformer_encoder(src=inp,
                                          mask=future_mask,
                                          src_key_padding_mask=src_key_padding_mask)

        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(output)

        # (batch_size, seq_length, d_model)
        output = output.permute(1, 0, 2)
        
        # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # zero-out padding embeddings
        # (batch_size, seq_length, d_model)
        output = output * padding_masks.unsqueeze(-1)

        # my addition turn into => [seq_len, d_model, batch_size]
        #output = output.permute(1, 2, 0)
        # squeeze the sequence length dimension
        output = output.reshape(output.size(0), -1)  # (batch_size, num_classes * horizon)
        #output = output.permute(1, 0)

        # (batch_size, seq_length, num_classes * horizon)
        output = self.output_layer(output)

        # (batch_size, seq_length, horizon, num_classes)
        #output = output.reshape(output.size(0), output.size(1), self.horizon, self.num_classes)

        return output
