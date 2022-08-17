import math
from collections import OrderedDict
from typing import Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor


def modified_sigmoid(x: Tensor) -> Tensor:
    return 2 * torch.sigmoid(x) ** 2.3 + 1e-7


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx}
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """

    def __init__(self, d_model, dropout=0.1, max_len=251):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, n_units=512, n_heads=8, n_hidden=2048, n_layers=6, dropout=0.1):
        super().__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = nn.TransformerEncoderLayer(n_units, n_heads, n_hidden, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.n_units = n_units

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        return output


class TransformerController(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        n_harmonics: int = 120,
        n_noise_filters: int = 100,
        n_units: int = 512,
        n_hidden: int = 1024,
        n_heads: int = 2,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_transform = nn.Linear(in_ch * 2, n_units)
        self.gru = nn.Transformer(n_units, n_heads, n_layers, n_layers, n_hidden, dropout)

        self.dense_harmonic = nn.Linear(n_units, n_harmonics)
        self.dense_loudness = nn.Linear(n_units, 1)
        self.dense_filter = nn.Linear(n_units, n_noise_filters)

    def forward(
        self, f0: Tensor, loudness: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]:
        # comes b, ch, t, linear needs b, t, ch, transformer needs t, b, ch
        features = torch.cat([f0, loudness], dim=1).transpose(1, 2)
        features = self.in_transform(features).transpose(0, 1)

        hidden = self.gru(features, features).transpose(0, 1)

        overtone_amplitudes = modified_sigmoid(self.dense_harmonic(hidden))
        master_amplitude = modified_sigmoid(self.dense_loudness(hidden))

        noise_distribution = self.dense_filter(hidden)
        noise_distribution = modified_sigmoid(noise_distribution)

        harm_controls = (
            f0,
            master_amplitude.transpose(1, 2),
            overtone_amplitudes.transpose(1, 2).unsqueeze(1),
        )
        noise_controls = noise_distribution.transpose(1, 2).unsqueeze(1)

        return harm_controls, noise_controls


class MLP(nn.Module):
    def __init__(
        self, n_input: int, n_units: int, n_layer: int, relu: Type[nn.Module] = nn.LeakyReLU
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_input = n_input
        self.n_units = n_units

        layers = [
            [
                nn.Linear(n_input, n_units),
                nn.LayerNorm(normalized_shape=n_units),
                relu(),
            ]
        ]

        for _ in range(1, n_layer):
            layers.append(
                [
                    nn.Linear(n_units, n_units),
                    nn.LayerNorm(normalized_shape=n_units),
                    relu(),
                ]
            )

        mlps = [nn.Sequential(*block) for block in layers]
        self.net = nn.Sequential(
            OrderedDict(zip((f"mlp_layer{i}" for i in range(1, n_layer + 1)), mlps))
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x


class Controller(nn.Module):
    def __init__(
        self,
        n_harmonics: int = 120,
        n_noise_filters: int = 100,
        decoder_mlp_units: int = 512,
        decoder_mlp_layers: int = 3,
        decoder_gru_units: int = 512,
        decoder_gru_layers: int = 1,
    ):
        super().__init__()

        self.mlp_f0 = MLP(n_input=1, n_units=decoder_mlp_units, n_layer=decoder_mlp_layers)
        self.mlp_loudness = MLP(n_input=1, n_units=decoder_mlp_units, n_layer=decoder_mlp_layers)

        self.num_mlp = 2

        self.gru = nn.GRU(
            input_size=self.num_mlp * decoder_mlp_units,
            hidden_size=decoder_gru_units,
            num_layers=decoder_gru_layers,
            batch_first=True,
        )

        self.mlp_gru = MLP(
            n_input=decoder_gru_units + self.num_mlp * decoder_mlp_units,
            n_units=decoder_mlp_units,
            n_layer=decoder_mlp_layers,
        )

        # one element for overall loudness
        self.dense_harmonic = nn.Linear(decoder_mlp_units, n_harmonics)
        self.dense_loudness = nn.Linear(decoder_mlp_units, 1)
        self.dense_filter = nn.Linear(decoder_mlp_units, n_noise_filters)

    def forward(
        self, f0: Tensor, loudness: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]:
        latent_f0 = self.mlp_f0(f0.transpose(1, 2))
        latent_loudness = self.mlp_loudness(loudness.transpose(1, 2))

        latent = torch.cat((latent_f0, latent_loudness), dim=-1)

        latent, _ = self.gru(latent)

        latent = torch.cat((latent, latent_f0, latent_loudness), dim=-1)
        latent = self.mlp_gru(latent)

        overtone_amplitudes = modified_sigmoid(self.dense_harmonic(latent))
        master_amplitude = modified_sigmoid(self.dense_loudness(latent))

        noise_distribution = self.dense_filter(latent)
        noise_distribution = modified_sigmoid(noise_distribution)

        harm_controls = (
            f0,
            master_amplitude.transpose(1, 2),
            overtone_amplitudes.transpose(1, 2).unsqueeze(1),
        )
        noise_controls = noise_distribution.transpose(1, 2).unsqueeze(1)

        return harm_controls, noise_controls
