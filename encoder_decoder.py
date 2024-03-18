from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from model import SEModule, PreEmphasis, FbankAug
from modules import ConvBlock1d, Squeeze, DepthWiseConv1d, AttentiveStatsPooling
from yc_utils import init_weights


# class SqueezeExcitation(nn.Module):
#     """
#     To tailor the SE module for speech processing tasks, Thien-pondt et al. [14] propose the frequency-wise
#     squeeze-excitation (fwSE) module, which aggregates global frequency information as attention weights
#     for all feature maps. The f -th element of of the frequency-wise mean statistics e∈ℝF is calculated as
#     https://arxiv.org/pdf/2104.02370.pdf
#     """
#     def __init__(self, channels, reduction=16):
#         super(SqueezeExcitation, self).__init__()
#
#         # Define architecture
#         self.squeeze1 = nn.AdaptiveAvgPool1d(1)
#         self.squeeze2 = nn.AdaptiveAvgPool2d(1)
#         self.excitation1 = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid(),
#         )
#         self.excitation2 = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.ReLU(),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid(),
#         )
#         self.excitation3 = nn.Sequential(
#             nn.Linear(channels, channels // (reduction / 2), bias=False),
#             nn.ReLU(),
#             nn.Linear(channels // (reduction / 2), channels, bias=False),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, inputs):
#         # [B, C, W] -> [B, C]   ===》 squeeze(inputs) [B,C,1], squeeze(-1)---》 [B,C]
#         squeezed1 = self.squeeze1(inputs).squeeze(-1)
#
#         # [B, C] -> [B, C]  ?? unsqueeze(-1)变为[B,C,1]
#         excited1 = self.excitation1(squeezed1).unsqueeze(-1)
# #         print(f"squeezed1.shape: {squeezed1.shape} excited1.shape: {excited1.shape}")
#
# #         squeezed2 = self.squeeze2(inputs)
# #         print(f"squeezed2.shape: {squeezed2.shape}")
# #         excited2 = self.excitation2(squeezed2)
# #         print(f"squeezed2.shape: {squeezed2.shape} excited2.shape: {excited2.shape}")
#
#         # [B, C] -> [B, C, W]
#         return inputs * excited1.expand_as(inputs) * excited1.expand_as(inputs)

class Encoder(nn.Module):
    """
    The TitaNet encoder starts with a prologue block, followed by a number
    of mega blocks and ends with an epilogue block; all blocks comprise
    convolutions, batch normalization, activation and dropout, while mega
    blocks are also equipped with residual connections and SE modules
    "TitaNet: Neural Model for speaker representation with 1D Depth-wise
    separable convolutions and global context", Kologuri et al.,
    https://arxiv.org/abs/2110.04410
    """

    def __init__(
            self,
            n_mels,
            n_mega_blocks,
            n_sub_blocks,
            hidden_size,
            output_size,
            mega_block_kernel_sizes,  # mega_block_kernel_size
            prolog_kernel_size=3,
            epilog_kernel_size=1,
            se_reduction=16,
            dropout=0.5,
            init_mode: Optional[str] = 'xavier_uniform',
    ):
        super(Encoder, self).__init__()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

        self.specaug = FbankAug()  # Spec augmentation

        # Define encoder as sequence of prolog, mega blocks and epilog
        self.prolog = ConvBlock1d(n_mels, hidden_size, prolog_kernel_size)
        self.mega_blocks = nn.Sequential(
            *[
                MegaBlock(
                    hidden_size,
                    hidden_size,
                    # mega_block_kernel_size,  # kernel_size
                    mega_block_kernel_sizes[_],  # mega_block_kernel_sizes = [3, 7, 11, 15]
                    n_sub_blocks,  # n_sub_blocks
                    se_reduction=se_reduction,
                    dropout=dropout,
                )
                # for _ in range(n_mega_blocks)
                for _ in range(len(mega_block_kernel_sizes))
            ]
        )
        self.epilog = ConvBlock1d(hidden_size, output_size, epilog_kernel_size)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, spectrograms, aug=True):
        """
        Given input spectrograms of shape [B, M, T], return encodings
        of shape [B, DE, T]
        B: batch size
        M: number of mel frequency bands : default = 40
        T: maximum number of time steps (frames)  : 帧数
        DE: encoding output size  : 输出尺寸
        H: hidden size : 隐藏层数
        """
        with torch.no_grad():
            x = self.torchfbank(spectrograms)+1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug:
                x = self.specaug(x)

        # [B, M, T] -> [B, H, T]
        prolog_outputs = self.prolog(x)

        # [B, H, T] -> [B, H, T]
        mega_blocks_outputs = self.mega_blocks(prolog_outputs)

        # [B, H, T] -> [B, DE, T]
        return self.epilog(mega_blocks_outputs)


class MegaBlock(nn.Module):
    """
    The TitaNet mega block, part of its encoder, comprises a sequence
    of sub-blocks, where each one contains a time-channel separable
    convolution followed by batch normalization, activation and dropout;
    the output of the sequence of sub-blocks is then processed by a SE
    module and merged with the initial input through a skip connection
    "TitaNet: Neural Model for speaker representation with 1D Depth-wise
    separable convolutions and global context", Kologuri et al.,
    https://arxiv.org/abs/2110.04410
    """

    def __init__(
            self,
            input_size,
            output_size,
            kernel_size,
            n_sub_blocks,
            se_reduction=16,
            dropout=0.5,
            se_res2block=False,
    ):
        super(MegaBlock, self).__init__()

        # Store attributes
        self.dropout = dropout

        # Define sub-blocks composed of depthwise convolutions
        channels = [input_size] + [output_size] * (n_sub_blocks - 1)  # [input_size, output_size, ..., output_size]
        se = SEModule
        self.sub_blocks = nn.Sequential(
            *[
                ConvBlock1d(
                    in_channels,  # [input_size, output_size, ..., output_size], size = n_sub_blocks - 1
                    out_channels,  # [output_size, ..., output_size], size = n_sub_blocks - 1
                    kernel_size,
                    activation="relu",
                    dropout=dropout,
                    depth_wise=True,
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ],
            DepthWiseConv1d(
                channels[-2],  # [input_size, output_size, ..., output_size], size = 1 + (n_sub_blocks-1)
                channels[-1],  # [output_size, ..., output_size], size = n_sub_blocks
                kernel_size=kernel_size,
                stride=1,
                dilation=1,
            ),
            nn.BatchNorm1d(channels[-1], eps=1e-3),  # 默认值 , momentum=0.1
            se(output_size, reduction=se_reduction)
        )

        # Define the final skip connection
        self.skip_connection = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=1),
            nn.BatchNorm1d(output_size, eps=1e-3),
        )

    def forward(self, prolog_outputs):
        """
        Given prolog outputs of shape [B, H, T], return
        a feature tensor of shape [B, H, T]
        B: batch size
        H: hidden size
        T: maximum number of time steps (frames)
        """
        # [B, H, T] -> [B, H, T]
        mega_block_outputs = self.skip_connection(prolog_outputs) + self.sub_blocks(
            prolog_outputs
        )
        return F.dropout(
            F.relu(mega_block_outputs), p=self.dropout, training=self.training
        )


def affine_layer(
        inp_shape, out_shape, learn_mean=True, affine_type='conv',
):
    if affine_type == 'conv':
        layer = nn.Sequential(
            # nn.BatchNorm1d(inp_shape, affine=True, track_running_stats=True),
            nn.Conv1d(inp_shape, out_shape, kernel_size=1),
        )

    else:
        layer = nn.Sequential(
            nn.Linear(inp_shape, out_shape),
            nn.BatchNorm1d(out_shape, affine=learn_mean, track_running_stats=True),
            nn.ReLU(),
        )

    return layer


class Decoder(nn.Module):
    """
    The TitaNet decoder computes intermediate time-independent features
    using an attentive statistics pooling layer and downsamples such
    representation using two linear layers, to obtain a fixed-size
    embedding vector first and class logits afterwards
    "TitaNet: Neural Model for speaker representation with 1D Depth-wise
    separable convolutions and global context", Kologuri et al.,
    https://arxiv.org/abs/2110.04410
    """

    def __init__(
            self,
            encoder_output_size,
            attention_hidden_size,
            embedding_size,
            pool_mode: str = 'simple',
            init_mode: Optional[str] = 'xavier_uniform',
    ):
        super(Decoder, self).__init__()

        # Define the attention/pooling layer
        self.affine_type = 'linear'  # 默认 pool 层后面跟随 linear 层
        if pool_mode == 'simple':
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                Squeeze(-1),
                # nn.Linear(encoder_output_size, encoder_output_size * 2),
                nn.Linear(encoder_output_size, encoder_output_size),
            )
        elif pool_mode == 'attention_stats':
            self.pool = nn.Sequential(
                AttentiveStatsPooling(encoder_output_size, attention_hidden_size),
                # nn.BatchNorm1d(encoder_output_size),  # 应该是已经设置成了3072，实际上回到了在通道维度有cat
                nn.BatchNorm1d(encoder_output_size * 2),
            )

        # Define the final classifier
        self.linear = affine_layer(encoder_output_size * 2, embedding_size, affine_type=self.affine_type)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, encodings):
        """
        Given encoder outputs of shape [B, DE, T], return a tensor
        of shape [B, E]
        B: batch size
        T: maximum number of time steps (frames)
        DE: encoding output size
        E: embedding size
        """
        # [B, DE, T] -> [B, DE * 2]
        pooled = self.pool(encodings)

        # [B, DE * 2] -> [B, E]
        embeddings = self.linear(pooled)
        if self.affine_type == 'conv':
            embeddings = embeddings.squeeze(-1)
        return embeddings
