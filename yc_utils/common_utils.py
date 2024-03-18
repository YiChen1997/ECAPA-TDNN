import os
import sys
import math
import wandb
import torch
import random
import numpy as np
from torch import nn
from torch.nn.init import _calculate_correct_fan
from typing import Callable, Iterable, List, Optional, Tuple

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_curve,
# )


def set_seed(seed):
    """
    Fix all possible sources of randomness
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"seed set as {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def optimizer_to(optimizer, device="cpu"):
    """
    Transfer the given optimizer to device
    """
    for param in optimizer.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for sub_param in param.values():
                if isinstance(sub_param, torch.Tensor):
                    sub_param.data = sub_param.data.to(device)
                    if sub_param._grad is not None:
                        sub_param._grad.data = sub_param._grad.data.to(device)
    return optimizer


def scheduler_to(scheduler, device="cpu"):
    """
    Transfer the given LR scheduler to device
    """
    for param in scheduler.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
    return scheduler


def chart_dependencies(model, n_mels=80, device="cpu"):
    """
    Use backprop to chart dependencies
    (see http://karpathy.github.io/2019/04/25/recipe/)
    """
    model.eval()
    batch_size, time_steps = random.randint(2, 10), random.randint(10, 100)
    inputs = torch.randn((batch_size, n_mels, time_steps)).to(device)
    inputs.requires_grad = True
    outputs = model(inputs)
    random_index = random.randint(0, batch_size)
    loss = outputs[random_index].sum()
    loss.backward()
    assert (
               torch.cat([inputs.grad[i] == 0 for i in range(batch_size) if i != random_index])
           ).all() and (
                   inputs.grad[random_index] != 0
           ).any(), f"Only index {random_index} should have non-zero gradients"


def init_wandb(api_key_file, project, entity, name=None, config=None, offline=False):
    """
    Return a new W&B run to be used for logging purposes

    :param offline 是否离线模式，不能连上wandb时选择此项
    """
    assert os.path.exists(api_key_file), "The given W&B API key file does not exist"
    api_key_value = open(api_key_file, "r").read().strip()
    os.environ["WANDB_API_KEY"] = api_key_value
    if offline:
        # 设置为离线模式
        os.environ["WANDB_MODE"] = "offline"
    # os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'
    # os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
    print(f"my wandb key is {api_key_value}")
    return wandb.init(
        name=name,
        project=project,
        entity=entity,
        config=config,
    )


class Struct:
    """
    Struct class, s.t. a nested dictionary is transformed
    into a nested object
    """

    def __init__(self, **entries):
        self.entries = entries
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__.update({k: Struct(**v)})
            else:
                self.__dict__.update({k: v})

    def get_true_key(self):
        """
        Return the only key in the Struct s.t. its value is True
        """
        true_types = [k for k, v in self.__dict__.items() if v == True]
        assert len(true_types) == 1
        return true_types[0]

    def get_true_keys(self):
        """
        Return all the keys in the Struct s.t. its value is True
        """
        return [k for k, v in self.__dict__.items() if v == True]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


def get_os() -> str:
    """
    返回系统版本，两种返回值，'win32'和 'linux'
    """
    return sys.platform


def tds_uniform_(tensor, mode='fan_in'):
    """
    Uniform Initialization from the paper [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2460.pdf)
    Normalized to -

    .. math::
        \\text{bound} = \\text{2} \\times \\sqrt{\\frac{1}{\\text{fan\\_mode}}}

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = 2.0  # sqrt(4.0) = 2
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def tds_normal_(tensor, mode='fan_in'):
    """
    Normal Initialization from the paper [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2460.pdf)
    Normalized to -

    .. math::
        \\text{bound} = \\text{2} \\times \\sqrt{\\frac{1}{\\text{fan\\_mode}}}

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = 2.0
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.normal_(0.0, bound)


def init_weights(m, mode: Optional[str] = 'xavier_uniform'):
    """
    Initialize the weights of a 网络
    Args:
        m: 要初始化的模型
        mode: 默认使用 xavier_uniform 进行初始化

    Returns:

    """
    # if isinstance(m, MaskedConv1d):
    #     init_weights(m.conv, mode)
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode is not None:
            if mode == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif mode == 'xavier_normal':
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif mode == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif mode == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif mode == 'tds_uniform':
                tds_uniform_(m.weight)
            elif mode == 'tds_normal':
                tds_normal_(m.weight)
            else:
                raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
