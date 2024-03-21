from typing import Optional
import torch
import torchaudio
from torch import nn

from campp import CAMPPlus
from model import PreEmphasis, FbankAug
from yc_utils import init_weights


class CamEncoder(nn.Module):
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
            init_mode: Optional[str] = 'xavier_uniform',
    ):
        super(CamEncoder, self).__init__()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

        self.specaug = FbankAug()  # Spec augmentation
        self.campp = CAMPPlus(feat_dim=n_mels)

        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, spectrograms, aug=False):
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
        return self.campp(x)
