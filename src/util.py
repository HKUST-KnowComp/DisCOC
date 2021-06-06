import json
import math
import os
import re
import numpy as np
import torch as th
import torch.nn as nn
from argparse import Namespace
from collections import namedtuple
from function import *


LEAKY_RELU_A = 1/5.5
WORD_EMB_MEAN = 0.0
WORD_EMB_STD = 1.0
TYPE_EMB_MEAN = 0.0
TYPE_EMB_STD = 0.02
URL_REGEX = re.compile(
    r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))',
    re.IGNORECASE
)


def clean_sentence_for_parsing(text):
    # only consider ascii
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # replace \n
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n+", " ", text)

    # replace ref
    text = re.sub(r"<ref(.*?)>", "<ref>", text)

    # replace url
    text = re.sub(URL_REGEX, "<url>", text)
    text = re.sub(r"<url>[\(\)\[\]]*<url>", "<url>", text)

    return text

def str2value(x):
    try:
        return eval(x)
    except:
        return x


def str2bool(x):
    x = x.lower()
    return x == "true" or x == "yes"


def str2list(x):
    results = []
    for x in x.split(","):
        x = x.strip()
        if x == "" or x == "null":
            continue
        try:
            x = str2value(x)
        except:
            pass
        results.append(x)
    return results


def load_config(path, as_dict=True):
    with open(path, "r") as f:
        config = json.load(f)
        if not as_dict:
            config = namedtuple("config", config.keys())(*config.values())
    return config


def save_config(config, path):
    if isinstance(config, dict):
        pass
    elif isinstance(config, Namespace):
        config = vars(config)
    else:
        try:
            config = config._as_dict()
        except BaseException as e:
            raise e

    with open(path, "w") as f:
        json.dump(config, f)


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray, th.Tensor)):
            return obj.tolist()
        else:
            return super(TensorEncoder, self).default(obj)


def load_results(path):
    with open(path, "w") as f:
        results = json.load(f)
    return results


def save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, cls=TensorEncoder)


def calculate_gain(activation):
    if isinstance(activation, str):
        if activation in ["none"]:
            nonlinearity = "linear"
        elif activation in ["relu", "relu6", "elu", "selu", "celu", "gelu"]:
            nonlinearity = "relu"
        elif activation in ["leaky_relu", "prelu"]:
            nonlinearity = "leaky_relu"
        elif activation in ["softmax"]:
            nonlinearity = "sigmoid"
        elif activation in ["sigmoid", "tanh"]:
            nonlinearity = activation
        else:
            raise NotImplementedError
    elif isinstance(activation, nn.Module):
        if isinstance(activation, Identity):
            nonlinearity = "linear"
        elif isinstance(activation, (ReLU, ReLU6, ELU, SELU, CELU, GELU)):
            nonlinearity = "relu"
        elif isinstance(activation, (LeakyReLU, PReLU)):
            nonlinearity = "leaky_relu"
        elif isinstance(activation, Softmax):
            nonlinearity = "sigmoid"
        elif isinstance(activation, Sigmoid):
            nonlinearity = "sigmoid"
        elif isinstance(activation, Tanh):
            nonlinearity = "tanh"
        else:
            raise NotImplementedError
    else:
        raise ValueError

    return nn.init.calculate_gain(nonlinearity, LEAKY_RELU_A)


def calculate_fan_in_and_fan_out(x):
    if x.dim() < 2:
        x = x.unsqueeze(-1)
    num_input_fmaps = x.size(1)
    num_output_fmaps = x.size(0)
    receptive_field_size = 1
    if x.dim() > 2:
        receptive_field_size = x[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def zero_init(x, gain=1.0):
    return nn.init.zeros_(x)


def xavier_uniform_init(x, gain=1.0):
    fan_in, fan_out = calculate_fan_in_and_fan_out(x)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = 1.7320508075688772 * std

    return nn.init.uniform_(x, -a, a)


def kaiming_normal_init(x, gain=1.0):
    fan_in, fan_out = calculate_fan_in_and_fan_out(x)
    std = gain / math.sqrt(fan_in)
    return nn.init.normal_(x, 0, std)


def orthogonal_init(x, gain=1.0):
    return nn.init.orthogonal_(x, gain=1.0)


def equivariant_init(x, gain=1.0):
    with th.no_grad():
        x_size = tuple(x.size())
        if len(x_size) == 1:
            kaiming_normal_init(x, gain=gain)
        elif len(x_size) == 2:
            kaiming_normal_init(x[0], gain=gain)
            vec = x[0]
            for i in range(1, x.size(0)):
                x[i].data.copy_(th.roll(vec, i, 0))
        else:
            x = x.view(x_size[:-2] + (-1,))
            equivariant_init(x, gain=gain)
            x = x.view(x_size)
    return x


def identity_init(x, gain=1.0):
    with th.no_grad():
        x_size = tuple(x.size())
        if len(x_size) == 1:
            fan_in, fan_out = calculate_fan_in_and_fan_out(x)
            std = gain * (2.0 / float(fan_in + fan_out)) * 0.2888
            nn.init.ones_(x)
            x += th.randn_like(x) * std
        elif len(x_size) == 2:
            fan_in, fan_out = calculate_fan_in_and_fan_out(x)
            std = gain * (2.0 / float(fan_in + fan_out)) * 0.2888
            nn.init.eye_(x)
            x += th.randn_like(x) * std
        else:
            x = x.view(x_size[:-2] + (-1,))
            identity_init(x, gain=gain)
            x = x.view(x_size)
    return x


def init_weight(x, activation="none", init="uniform"):
    gain = calculate_gain(activation)
    if init == "zero":
        init_func = zero_init
    elif init == "identity":
        init_func = identity_init
    elif init == "uniform":
        init_func = xavier_uniform_init
    elif init == "normal":
        init_func = kaiming_normal_init
    elif init == "orthogonal":
        init_func = orthogonal_init
    elif init == "equivariant":
        init_func = equivariant_init
    else:
        raise ValueError("init=%s is not supported now." % (init))

    if isinstance(x, th.Tensor):
        init_func(x, gain=gain)
    elif isinstance(x, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        init_func(x.weight, gain=gain)
        if hasattr(x, "bias") and x.bias is not None:
            zero_init(x.bias)
    elif isinstance(x, nn.Embedding):
        with th.no_grad():
            if init == "uniform":
                nn.init.uniform_(x.weight, -WORD_EMB_STD/0.2888, WORD_EMB_STD/0.2888)
                x.weight += WORD_EMB_MEAN
            elif init == "normal":
                nn.init.normal_(x.weight, WORD_EMB_MEAN, WORD_EMB_STD)
            elif init == "orthogonal":
                nn.init.orthogonal_(x.weight, gain=math.sqrt(calculate_fan_in_and_fan_out(x.weight)[0]) * WORD_EMB_STD)
            elif init == "identity":
                nn.init.eye_(x.weight)
            elif init == "equivariant":
                nn.init.normal_(x.weight[0], WORD_EMB_MEAN, WORD_EMB_STD)
                vec = x.weight[0]
                for i in range(1, x.weight.size(0)):
                    x.weight[i].data.copy_(th.roll(vec, i, 0))
            if x.padding_idx is not None:
                x.weight[x.padding_idx].fill_(0)
    elif isinstance(x, nn.RNNBase):
        for layer_weights in x._all_weights:
            for w in layer_weights:
                if "weight" in w:
                    init_func(getattr(x, w))
                elif "bias" in w:
                    zero_init(getattr(x, w))
    elif isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
        nn.init.ones_(x.weight)
        nn.init.zeros_(x.bias)



def batch_convert_list_to_tensor(batch_list, max_seq_len=-1, pad_id=0, pre_pad=False):
    batch_tensor = [th.tensor(v) for v in batch_list]
    return batch_convert_tensor_to_tensor(batch_tensor, max_seq_len=max_seq_len, pad_id=pad_id, pre_pad=pre_pad)


def batch_convert_tensor_to_tensor(batch_tensor, max_seq_len=-1, pad_id=0, pre_pad=False):
    batch_lens = [len(v) for v in batch_tensor]
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)

    result = th.empty(
        [len(batch_tensor), max_seq_len] + list(batch_tensor[0].size())[1:],
        dtype=batch_tensor[0].dtype,
        device=batch_tensor[0].device
    ).fill_(pad_id)
    for i, t in enumerate(batch_tensor):
        len_t = batch_lens[i]
        if len_t < max_seq_len:
            if pre_pad:
                result[i, -len_t:].data.copy_(t)
            else:
                result[i, :len_t].data.copy_(t)
        elif len_t == max_seq_len:
            result[i].data.copy_(t)
        else:
            result[i].data.copy_(t[:max_seq_len])
    return result


def batch_convert_len_to_mask(batch_lens, max_seq_len=-1, pre_pad=False):
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)
    mask = th.ones(
        (len(batch_lens), max_seq_len),
        dtype=th.bool,
        device=batch_lens[0].device if isinstance(batch_lens[0], th.Tensor) else th.device("cpu")
    )
    if pre_pad:
        for i, l in enumerate(batch_lens):
            mask[i, :-l].fill_(0)
    else:
        for i, l in enumerate(batch_lens):
            mask[i, l:].fill_(0)
    return mask


def batch_convert_mask_to_start_and_end(mask):
    cumsum = mask.cumsum(dim=-1) * 2
    start_indices = cumsum.masked_fill(mask == 0, mask.size(-1)).min(dim=-1)[1]
    end_indices = cumsum.max(dim=-1)[1]

    return start_indices, end_indices


def split_ids(x_ids):
    if x_ids[0] == x_ids[-1]:
        return th.LongTensor([x_ids.size(0)]).to(x_ids.device)
    diff = th.roll(x_ids, -1, 0) - x_ids
    return th.masked_select(th.arange(1, x_ids.size(0) + 1, device=x_ids.device), diff.bool())


def change_dropout_rate(model, dropout):
    for name, child in model.named_children():
        if isinstance(child, nn.Dropout):
            child.p = dropout
        change_dropout_rate(child, dropout)

