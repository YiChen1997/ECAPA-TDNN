"""
    包含：
    - 卷积1D
    - 深度可分离卷积1D
    - 基于深度可分离卷积1D的卷积块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Conv1d):
    """
    1D convolutional layer with "same" padding (no downsampling),
    that is also compatible with strides > 1
    """

    def __init__(self, *args, **kwargs):
        super(Conv1dSamePadding, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        """
        Given an input of size [B, CI, WI], return an output
        [B, CO, WO], where WO = [CI + 2P - K - (K - 1) * (D - 1)] / S + 1,
        by computing P on-the-fly ay forward time

        B: batch size
        CI: input channels
        WI: input width
        CO: output channels
        WO: output width
        P: padding
        K: kernel size
        D: dilation
        S: stride
        """
        padding = (
                          self.stride[0] * (inputs.shape[-1] - 1)
                          - inputs.shape[-1]
                          + self.kernel_size[0]
                          + (self.dilation[0] - 1) * (self.kernel_size[0] - 1)
                  ) // 2
        return self._conv_forward(
            F.pad(inputs, (padding, padding)),
            self.weight,
            self.bias,
        )


class DepthWiseConv1d(nn.Module):
    """
    Compute a depth-wise separable convolution, by performing
    a depth-wise convolution followed by a point-wise convolution

    "Xception: Deep Learning with Depthwise Separable Convolutions",
    Chollet, https://arxiv.org/abs/1610.02357
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            bias=True,
            device=None,
            dtype=None,
    ):
        super(DepthWiseConv1d, self).__init__()
        self.conv = nn.Sequential(
            Conv1dSamePadding(  # 深度可分离卷积是让output_channels与input_channels相同，同时groups等于input_channels
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
                bias=bias,
                device=device,
                dtype=dtype,
            ),  # 这时输出通道与输入通道相同，如果需要改变输出通道数，则再跟一个kernel等于1的1D卷积即可。
            Conv1dSamePadding(
                in_channels, out_channels, kernel_size=1, device=device, dtype=dtype
            ),
        )

    def forward(self, inputs):
        """
        Given an input of size [B, CI, WI], return an output
        [B, CO, WO], where CO is given as a parameter and WO
        depends on the convolution operation attributes

        B: batch size
        CI: input channels
        WI: input width
        CO: output channels
        WO: output width
        """
        return self.conv(inputs)


class ConvBlock1d(nn.Module):
    """
    Standard convolution, normalization, activation block
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            activation="relu",
            dropout=0,
            depth_wise=False,
    ):
        super(ConvBlock1d, self).__init__()
        assert activation is None or activation in (
            "relu",
            "tanh",
        ), "Incompatible activation function"

        # Define architecture
        conv_module = DepthWiseConv1d if depth_wise else Conv1dSamePadding
        modules = [
            conv_module(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels, eps=1e-3),
        ]
        if activation is not None:
            modules += [nn.ReLU() if activation == "relu" else nn.Tanh()]
        if dropout > 0:
            modules += [nn.Dropout(p=dropout)]
        self.conv_block = nn.Sequential(*modules)

    def forward(self, inputs):
        """
        Given an input of size [B, CI, WI], return an output
        [B, CO, WO], where CO is given as a parameter and WO
        depends on the convolution operation attributes

        B: batch size
        CI: input channels
        WI: input width
        CO: output channels
        WO: output width
        """
        return self.conv_block(inputs)


class Squeeze(nn.Module):
    """
    Remove dimensions of size 1 from the input tensor
    """

    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return inputs.squeeze(self.dim)

class AttentiveStatsPooling(nn.Module):
    """
    The attentive statistics pooling layer uses an attention
    mechanism to give different weights to different frames and
    generates not only weighted means but also weighted variances,
    to form utterance-level features from frame-level features
    "Attentive Statistics Pooling for Deep Speaker Embedding",
    Okabe et al., https://arxiv.org/abs/1803.10963
    """

    def __init__(self, input_size, hidden_size, eps=1e-6):
        super(AttentiveStatsPooling, self).__init__()

        # Store attributes
        self.eps = eps

        # Define architecture
        self.in_linear = nn.Linear(input_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, input_size)

    def forward(self, encodings):
        """
        Given encoder outputs of shape [B, DE, T], return
        pooled outputs of shape [B, DE * 2]
        B: batch size
        T: maximum number of time steps (frames)
        DE: encoding output size
        """
        # Compute a scalar score for each frame-level feature
        # [B, DE, T] -> [B, DE, T]
        energies = self.out_linear(
            torch.tanh(self.in_linear(encodings.transpose(1, 2)))
        ).transpose(1, 2)

        # Normalize scores over all frames by a softmax function
        # [B, DE, T] -> [B, DE, T]
        alphas = torch.softmax(energies, dim=2)

        # Compute mean vector weighted by normalized scores
        # [B, DE, T] -> [B, DE]
        means = torch.sum(alphas * encodings, dim=2)

        # Compute std vector weighted by normalized scores
        # [B, DE, T] -> [B, DE]
        residuals = torch.sum(alphas * encodings ** 2, dim=2) - means ** 2
        stds = torch.sqrt(residuals.clamp(min=self.eps))

        # Concatenate mean and std vectors to produce
        # utterance-level features
        # [[B, DE]; [B, DE]] -> [B, DE * 2]
        return torch.cat([means, stds], dim=1)
