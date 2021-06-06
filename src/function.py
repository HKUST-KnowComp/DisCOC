import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.functional import sigmoid
from torch.nn.modules.activation import Sigmoid

from torch.nn.functional import softmax
from torch.nn.modules.activation import Softmax

from torch.nn.functional import tanh
from torch.nn.modules.activation import Tanh

from torch.nn.functional import relu
from torch.nn.modules.activation import ReLU

from torch.nn.functional import relu6
from torch.nn.modules.activation import ReLU6

from torch.nn.functional import leaky_relu
from torch.nn.modules.activation import LeakyReLU

from torch.nn.functional import prelu
from torch.nn.modules.activation import PReLU


LEAKY_RELU_A = 1 / 5.5
PI = 3.141592653589793
EPS = 1e-8
INF = 1e30
_INF = -1e30


class Identity(nn.Module):
    def forward(self, x):
        return x

try:
    from torch.nn.functional import elu
except ImportError:

    def elu(x, alpha=1.0, inplace=False):
        if inplace:
            neg_x = th.clamp(alpha * (th.exp(x.clone()) - 1), max=0)
            th.clamp_(x, min=0)
            x += neg_x
            return x
        else:
            return th.clamp(x, min=0) + th.clamp(alpha * (th.exp(x) - 1), max=0)


try:
    from torch.nn.modules.activation import ELU
except ImportError:

    class ELU(nn.Module):
        def __init__(self, alpha=1.0, inplace=False):
            super(ELU, self).__init__()
            self.alpha = alpha
            self.inplace = inplace

        def forward(self, x):
            return elu(x, self.alpha, self.inplace)

        def extra_repr(self):
            return "alpha={}{}".format(self.alpha, ", inplace=True" if self.inplace else "")


"""
Gaussian Error Linear Units (GELUs)
Dan Hendrycks, Kevin Gimpel
https://arxiv.org/abs/1606.08415
"""
try:
    from torch.nn.functional import gelu
except ImportError:

    def gelu(x):
        return 0.5 * x * (1 + th.tanh(math.sqrt(2 / PI) * (x + 0.044715 * th.pow(x, 3))))


try:
    from torch.nn.modules.activation import GELU
except ImportError:

    class GELU(nn.Module):
        def __init__(self):
            super(GELU, self).__init__()

        def forward(self, x):
            return gelu(x)


"""
Self-Normalizing Neural Networks
Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter
https://arxiv.org/abs/1706.02515
"""
try:
    from torch.nn.functional import selu
except ImportError:

    def selu(x, inplace=False):
        if inplace:
            neg_x = th.clamp(1.0507009873554804934193349852946 * (th.exp(x.clone()) - 1), max=0)
            th.clamp_(x, min=0)
            x += neg_x
            x *= 1.6732632423543772848170429916717
            return x
        else:
            return 1.6732632423543772848170429916717 * (
                th.clamp(x, min=0) + th.clamp(1.0507009873554804934193349852946 * (th.exp(x) - 1), max=0)
            )


try:
    from torch.nn.modules.activation import SELU
except ImportError:

    class SELU(nn.Module):
        def __init__(self, inplace=False):
            super(SELU, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            return selu(x, self.inplace)

        def extra_repr(self):
            return "inplace=True" if self.inplace else ""


"""
Continuously Differentiable Exponential Linear Units
Jonathan T. Barron
https://arxiv.org/abs/1704.07483
"""
try:
    from torch.nn.functional import celu
except ImportError:

    def celu(x, alpha=1.0, inplace=False):
        if inplace:
            neg_x = th.clamp(alpha * (th.exp(x.clone() / alpha) - 1), max=0)
            th.clamp_(x, min=0)
            x += neg_x
            return x
        else:
            return th.clamp(x, min=0) + th.clamp(alpha * (th.exp(x / alpha) - 1), max=0)


try:
    from torch.nn.modules.activation import CELU
except ImportError:

    class CELU(nn.Module):
        def __init__(self, alpha=1.0, inplace=False):
            super(CELU, self).__init__()
            self.alpha = alpha
            self.inplace = inplace

        def forward(self, x):
            return celu(x, self.alpha)

        def extra_repr(self):
            return "alpha={}{}".format(self.alpha, ", inplace=True" if self.inplace else "")


"""
Continuously Differentiable Exponential Linear Units
Jonathan T. Barron
https://arxiv.org/abs/1704.07483
"""
try:
    from torch.nn.functional import rrelu
except ImportError:

    def rrelu(x, lower=1 / 8, upper=1 / 3, training=False, inplace=False):
        a = th.rand(1, device=x.device) * (1 / 3 - 1 / 8) + 1 / 8
        if inplace:
            neg_x = th.clamp(a * x.clone(), max=0)
            th.clamp_(x, min=0)
            x += neg_x
            return x
        else:
            return th.clamp(x, min=0) + th.clamp(a * x, max=0)


try:
    from torch.nn.modules.activation import RReLU
except ImportError:

    class RReLU(nn.Module):
        def __init__(self, lower=1 / 8, upper=1 / 3, inplace=False):
            super(RReLU, self).__init__()
            self.lower = lower
            self.upper = upper
            self.inplace = inplace

        def forward(self, x):
            return rrelu(x, self.lower, self.upper, self.training, self.inplace)

        def extra_repr(self):
            return "lower={}, upper={}{}".format(self.lower, self.upper, ", inplace=True" if self.inplace else "")


supported_activations = {
    "none": Identity(),
    "softmax": Softmax(dim=-1),
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
    "relu": ReLU(),
    "relu6": ReLU6(),
    "leaky_relu": LeakyReLU(negative_slope=LEAKY_RELU_A),
    "prelu": PReLU(init=LEAKY_RELU_A),
    "elu": ELU(),
    "celu": CELU(),
    "selu": SELU(),
    "gelu": GELU()
}


def map_activation_str_to_layer(activation, **kw):
    if activation not in supported_activations:
        print(activation)
        raise NotImplementedError

    act = supported_activations[activation]
    for k, v in kw.items():
        if hasattr(act, k):
            try:
                setattr(act, k, v)
            except:
                pass
    return act


def masked_max(x, mask, dim, keepdim=False):
    if mask is None:
        return x

    while mask.dim() < x.dim():
        mask = mask.unsqueeze(1)
    masked_x = x.masked_fill(mask == 0, _INF)
    max_value, _ = masked_x.max(dim=dim, keepdim=keepdim)
    return max_value


def masked_mean(x, mask, dim, keepdim=False):
    if mask is None:
        return x

    while mask.dim() < x.dim():
        mask = mask.unsqueeze(1)
    masked_x = x.masked_fill(mask == 0, 0.0)
    value_sum = th.sum(masked_x, dim=dim, keepdim=keepdim)
    value_count = th.sum(mask.float(), dim=dim, keepdim=keepdim)
    return value_sum / value_count.clamp(min=EPS)


def masked_softmax(x, mask, dim=-1):
    if mask is None:
        return F.softmax(x, dim=dim)

    while mask.dim() < x.dim():
        mask = mask.unsqueeze(1)
    masked_x = x.masked_fill(mask == 0, _INF)
    return F.softmax(masked_x, dim=dim)


def multi_perspective_match(x1, x2, weight):
    assert x1.size(0) == x2.size(0)
    assert weight.size(1) == x1.size(2)

    # (batch, seq_len, 1)
    similarity_single = F.cosine_similarity(x1, x2, 2).unsqueeze(2)

    # (1, 1, num_perspectives, hidden_size)
    weight = weight.unsqueeze(0).unsqueeze(0)

    # (batch, seq_len, num_perspectives, hidden_size)
    x1 = weight * x1.unsqueeze(2)
    x2 = weight * x2.unsqueeze(2)

    similarity_multi = F.cosine_similarity(x1, x2, 3)

    return similarity_single, similarity_multi


def multi_perspective_match_pairwise(x1, x2, weight):
    num_perspectives = weight.size(0)

    # (1, num_perspectives, 1, hidden_size)
    weight = weight.unsqueeze(0).unsqueeze(2)

    # (batch, num_perspectives, seq_len*, hidden_size)
    x1 = weight * x1.unsqueeze(1).expand(-1, num_perspectives, -1, -1)
    x2 = weight * x2.unsqueeze(1).expand(-1, num_perspectives, -1, -1)

    # (batch, num_perspectives, seq_len*, 1)
    x1_norm = x1.norm(p=2, dim=3, keepdim=True)
    x2_norm = x2.norm(p=2, dim=3, keepdim=True)

    # (batch, num_perspectives, seq_len1, seq_len2)
    mul_result = th.matmul(x1, x2.transpose(2, 3))
    norm_value = x1_norm * x2_norm.transpose(2, 3)

    # (batch, seq_len1, seq_len2, num_perspectives)
    return (mul_result / norm_value.clamp(min=EPS)).permute(0, 2, 3, 1)
