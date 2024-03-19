"""
    包含：
    - 卷积1D
    - 深度可分离卷积1D
    - 基于深度可分离卷积1D的卷积块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from typing import List

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


class AttentivePoolLayer(nn.Module):
    """
    Attention pooling layer for pooling speaker embeddings
    Reference: ECAPA-TDNN Embeddings for Speaker Diarization (https://arxiv.org/pdf/2104.01466.pdf)
    inputs:
        inp_filters: input feature channel length from encoder
        attention_channels: intermediate attention channel size
        kernel_size: kernel_size for TDNN and attention conv1d layers (default: 1)
        dilation: dilation size for TDNN and attention conv1d layers  (default: 1)
    """

    def __init__(
        self,
        inp_filters: int,
        attention_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        eps: float = 1e-10,
    ):
        super().__init__()

        self.feat_in = 2 * inp_filters

        self.attention_layer = nn.Sequential(
            TDNNModule(inp_filters * 3, attention_channels, kernel_size=kernel_size, dilation=dilation),
            nn.Tanh(),
            nn.Conv1d(
                in_channels=attention_channels, out_channels=inp_filters, kernel_size=kernel_size, dilation=dilation,
            ),
        )
        self.eps = eps

    def forward(self, x, length=None):
        max_len = x.size(2)

        if length is None:
            length = torch.ones(x.shape[0], device=x.device)

        mask, num_values = lens_to_mask(length, max_len=max_len, device=x.device)

        # encoder statistics
        mean, std = get_statistics_with_mask(x, mask / num_values)
        mean = mean.unsqueeze(2).repeat(1, 1, max_len)
        std = std.unsqueeze(2).repeat(1, 1, max_len)
        attn = torch.cat([x, mean, std], dim=1)

        # attention statistics
        attn = self.attention_layer(attn)  # attention pass
        attn = attn.masked_fill(mask == 0, -inf)
        alpha = F.softmax(attn, dim=2)  # attention values, α
        mu, sg = get_statistics_with_mask(x, alpha)  # µ and ∑

        # gather
        return torch.cat((mu, sg), dim=1).unsqueeze(2)


class TDNNModule(nn.Module):
    """
    Time Delayed Neural Module (TDNN) - 1D
    input:
        inp_filters: input filter channels for conv layer
        out_filters: output filter channels for conv layer
        kernel_size: kernel weight size for conv layer
        dilation: dilation for conv layer
        stride: stride for conv layer
        padding: padding for conv layer (default None: chooses padding value such that input and output feature shape matches)
    output:
        tdnn layer output
    """

    def __init__(
        self,
        inp_filters: int,
        out_filters: int,
        kernel_size: int = 1,
        dilation: int = 1,
        stride: int = 1,
        padding: int = None,
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size, stride=stride, dilation=dilation)

        self.conv_layer = nn.Conv1d(
            in_channels=inp_filters,
            out_channels=out_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_filters)

    def forward(self, x, length=None):
        x = self.conv_layer(x)
        x = self.activation(x)
        return self.bn(x)


def lens_to_mask(lens: List[int], max_len: int, device: str = None):
    """
    outputs masking labels for list of lengths of audio features, with max length of any
    mask as max_len
    input:
        lens: list of lens
        max_len: max length of any audio feature
    output:
        mask: masked labels
        num_values: sum of mask values for each feature (useful for computing statistics later)
    """
    lens_mat = torch.arange(max_len).to(device)
    mask = lens_mat[:max_len].unsqueeze(0) < lens.unsqueeze(1)
    mask = mask.unsqueeze(1)
    num_values = torch.sum(mask, dim=2, keepdim=True)
    return mask, num_values


def get_statistics_with_mask(x: torch.Tensor, m: torch.Tensor, dim: int = 2, eps: float = 1e-10):
    """
    compute mean and standard deviation of input(x) provided with its masking labels (m)
    input:
        x: feature input
        m: averaged mask labels
    output:
        mean: mean of input features
        std: stadard deviation of input features
    """
    mean = torch.sum((m * x), dim=dim)
    std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
    return mean, std


def get_same_padding(kernel_size, stride, dilation) -> int:
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2

