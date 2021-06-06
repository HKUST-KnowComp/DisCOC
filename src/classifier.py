import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dataset import CONTEXT, TEXT, CLS
from encoder import *
from function import map_activation_str_to_layer
from layer import *
from util import *


INF = 1e30
_INF = -1e30


def process_indices(sent_ids):
    if sent_ids.dim() == 1:
        indices = split_ids(sent_ids)
    elif sent_ids.dim() == 2:
        indices = split_ids(sent_ids[0])
    else:
        raise ValueError("Error: sent_ids.dim() != 1 or 2.")
    return indices


def process_type_ids(x1_indices, x2_indices, method="bert"):
    x1_len = x1_indices[-1].item()
    x2_len = x2_indices[-1].item()

    # BERT (Devlin et al., 2019)
    # only two segment ids
    if method == "bert" or method == "flat":
        x1_type_ids = th.zeros((x1_len, ), dtype=th.long, device=x1_indices.device)
        x2_type_ids = th.ones((x2_len, ), dtype=th.long, device=x2_indices.device)

    # XLNet (Yang et al., 2019)
    # each segment has an unique id
    elif method == "xlnet" or method == "segmented":
        x1_type_ids = th.zeros((x1_len, ), dtype=th.long, device=x1_indices.device)
        for i in range(1, len(x1_indices)):
            x1_type_ids[x1_indices[i-1]:x1_indices[i]].fill_(i)

        bias = len(x1_indices)
        x2_type_ids = th.empty((x2_len, ), dtype=th.long, device=x2_indices.device).fill_(bias)
        for i in range(1, len(x2_indices)):
            x2_type_ids[x2_indices[i-1]:x2_indices[i]].fill_(i + bias)

    # BERTSum (Liu et al., 2019)
    # use segment ids in turn
    # 0/1: context
    # 0/1: text
    elif method == "bert-sum" or method == "naive-interval":
        # make sure the last context as 1
        bias = len(x1_indices) % 2
        x1_type_ids = th.zeros((x1_len, ), dtype=th.long, device=x1_indices.device).fill_(bias)
        for i in range(1, len(x1_indices)):
            x1_type_ids[x1_indices[i-1]:x1_indices[i]].fill_((i + bias) % 2)

        x2_type_ids = th.zeros((x2_len, ), dtype=th.long, device=x2_indices.device).fill_(2)
        # make sure the first text as 2
        for i in range(1, len(x2_indices)):
            x2_type_ids[x2_indices[i-1]:x2_indices[i]].fill_(i % 2)

    # BMGF-RoBERTa (Liu et al., 2020)
    # use 0 for previous context
    # 0/1: context
    # 2/3: text
    elif method == "bmgf-roberta" or method == "interval":
        # make sure the last context as 1
        bias = len(x1_indices) % 2
        x1_type_ids = th.zeros((x1_len, ), dtype=th.long, device=x1_indices.device).fill_(bias)
        for i in range(1, len(x1_indices)):
            x1_type_ids[x1_indices[i-1]:x1_indices[i]].fill_((i + bias) % 2)

        x2_type_ids = th.zeros((x2_len, ), dtype=th.long, device=x2_indices.device).fill_(2)
        # make sure the first text as 2
        for i in range(1, len(x2_indices)):
            x2_type_ids[x2_indices[i-1]:x2_indices[i]].fill_(2 + i % 2)

    else:
        raise NotImplementedError(
            "Error: the method of process_type_ids should be \
            \"bert (flat)\", \"xlnet (segmented)\", \"bert-sum (naive-interval)\", or \"bmgf-roberta (interval)\"."
        )

    return x1_type_ids, x2_type_ids


#################################################################
########################## Flatten Model ########################
#################################################################


class FlatModel(nn.Module):
    def __init__(self, **kw):
        super(FlatModel, self).__init__()
        max_num_text = kw.get("max_num_text", 1)
        max_num_context = kw.get("max_num_context", 1)
        encoder = kw.get("encoder", "roberta")
        dropout = kw.get("dropout", 0.0)

        self.max_num_context = max_num_context
        self.max_num_text = max_num_text
        self.drop = nn.Dropout(dropout, inplace=False)
        if encoder == "bert":
            self.encoder = BertEncoder(num_segments=max_num_text+max_num_context+2, **kw)
        elif encoder == "albert":
            self.encoder = AlbertEncoder(num_segments=max_num_text+max_num_context+2, **kw)
        elif encoder == "roberta":
            self.encoder = RobertaEncoder(num_segments=max_num_text+max_num_context+2, **kw)
        elif encoder == "xlnet":
            self.encoder = XLNetEncoder(num_segments=max_num_text+max_num_context+2, **kw)
        elif encoder == "lstm":
            self.encoder = LSTMEncoder(num_segments=max_num_text+max_num_context+2, **kw)
        else:
            raise NotImplementedError("Error: encoder=%s is not supported now." % (encoder))
        self.create_layers(self.encoder.get_output_dim(), **kw)

    def create_layers(self, input_dim, **kw):
        max_num_text = kw.get("max_num_text", 1)
        max_num_context = kw.get("max_num_context", 1)
        max_len = kw.get("max_len", 512)
        hidden_dim = kw.get("hidden_dim", 128)
        add_matching = kw.get("add_matching", False)
        add_fusion = kw.get("add_fusion", False)
        add_conv = kw.get("add_conv", False)
        add_trans = kw.get("add_trans", False)
        add_gru = kw.get("add_gru", False)
        conv_filters = kw.get("conv_filters", 64)
        num_perspectives = kw.get("num_perspectives", 8)
        num_labels = kw.get("num_labels", 3)
        dropout = kw.get("dropout", 0.0)
        activation = kw.get("activation", "relu")

        dim = input_dim
        if add_matching:
            # bidirectional matching
            self.matching_layer = BiMpmMatching(hidden_dim=dim, num_perspectives=num_perspectives)
            dim = dim + self.matching_layer.get_output_dim() * 2
        else:
            self.register_parameter("matching_layer", None)
        if add_fusion:
            self.fusion_layer = DotAttention(
                query_dim=dim,
                key_dim=dim,
                value_dim=dim,
                hidden_dim=hidden_dim,
                num_heads=num_perspectives,
                scale=1 / hidden_dim**0.5,
                score_func="softmax",
                add_zero_attn=False,
                add_residual=False,
                add_gate=True,
                pre_lnorm=True,
                post_lnorm=False,
                dropout=dropout
            )
        else:
            self.register_parameter("fusion_layer", None)
        if add_conv:
            self.conv_layer = CnnHighway(
                input_dim=dim,
                filters=[(1, conv_filters), (2, conv_filters)],
                num_highway=1,
                activation=activation,
                layer_norm=False
            )
            dim = conv_filters * 2
        else:
            self.register_parameter("conv_layer", None)
        if add_trans:
            self.pos_emb = PositionEmbedding(
                input_dim=dim,
                max_len=(max_num_text+max_num_context+2),
                scale=INIT_EMB_STD
            )
            self.trans_layer = DotAttention(
                query_dim=dim,
                key_dim=dim,
                value_dim=dim,
                hidden_dim=hidden_dim,
                num_heads=num_perspectives,
                scale=1 / hidden_dim**0.5,
                score_func="softmax",
                add_zero_attn=False,
                add_residual=False,
                add_gate=True,
                pre_lnorm=True,
                post_lnorm=False,
                dropout=dropout
            )
        else:
            self.register_parameter("pos_emb", None)
            self.register_parameter("trans_layer", None)
        if add_gru:
            self.gru_layer = nn.GRU(
                input_size=dim,
                hidden_size=dim//2,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )
        else:
            self.register_parameter("gru_layer", None)
        self.fc_layer = MLP(
            input_dim=dim,
            hidden_dim=hidden_dim,
            output_dim=num_labels,
            num_mlp_layers=2,
            activation="none",
            norm_layer="batch_norm"
        )

        # init
        if self.gru_layer is not None:
            init_weight(self.gru_layer)

    def set_finetune(self, finetune):
        assert finetune in ["full", "layers", "last", "type", "none"]

        for param in self.parameters():
            param.requires_grad = True

        self.encoder.set_finetune(finetune)

        # fix word embeddings
        # if isinstance(self.encoder, LSTMEncoder):
        #     self.encoder.word_embeddings.weight.requires_grad = False
        # elif isinstance(self.encoder, XLNetEncoder):
        #     self.encoder.model.word_embedding.weight.requires_grad = False
        # else:
        #     self.encoder.model.embeddings.word_embeddings.weight.requires_grad = False

        # fix position embeddings
        # if isinstance(self.encoder, LSTMEncoder):
        #     self.position_embeddings.weight.requires_grad = False
        # elif isinstance(self.encoder, XLNetEncoder):
        #     pass
        # else:
        #     self.encoder.model.embeddings.position_embeddings.weight.requires_grad = False

    def load_pt(self, model_path, device=None):
        if device is None:
            device = th.device("cpu")
        own_dict = self.state_dict()
        state_dict = th.load(model_path, map_location=device)
        try:
            for name, param in state_dict.items():
                if "fc_layer" in name:
                    print("skip the fully connected layer")
                    continue
                if name not in own_dict:
                    continue
                if isinstance(param, nn.Parameter):
                    param = param.data
                if param.size() == own_dict[name].size():
                    own_dict[name].copy_(param)
            # self.load_state_dict(state_dict, strict=False)
        except BaseException as e:
            print(e)

    def encode(
        self,
        x1,
        x2,
        x1_mask=None,
        x2_mask=None,
        x1_sent_ids=None,
        x2_sent_ids=None,
        stance_logit=None,
        disco_logit=None
    ):
        if isinstance(self, nn.DataParallel):
            encoder = self.module.encoder
        else:
            encoder = self.encoder

        bsz, x1_len = x1.size()
        x2_len = x2.size(1)
        x_len = x1_len + x2_len

        if x1_mask is None:
            x1_mask = th.ones((bsz, x1_len), dtype=th.bool, device=x1.device)
        if x2_mask is None:
            x2_mask = th.ones((bsz, x2_len), dtype=th.bool, device=x2.device)

        if x1_sent_ids is not None:
            x1_indices = process_indices(x1_sent_ids)
        else:
            x1_sent_ids = th.zeros_like(x1)
            x1_indices = th.tensor([x1_len], dtype=th.long, device=x1.device)
        if x2_sent_ids is not None:
            x2_indices = process_indices(x2_sent_ids)
        else:
            x2_sent_ids = th.zeros_like(x2)
            x2_indices = th.tensor([x2_len], dtype=th.long, device=x2.device)

        x1_type_ids, x2_type_ids = process_type_ids(x1_indices, x2_indices, method="flat")
        x1_indices, x2_indices = x1_indices.tolist(), x2_indices.tolist()

        x = th.cat([x1, x2], dim=1)
        mask = th.cat([x1_mask, x2_mask], dim=1)
        # pos_ids = th.cumsum(mask, dim=1).masked_fill(th.logical_not(mask), 0)
        pos_ids = None
        sent_ids = th.cat([x1_sent_ids, x2_sent_ids+x1_sent_ids[:, -1:]+1], dim=1)
        type_ids = th.cat([x1_type_ids, x2_type_ids], dim=0).unsqueeze(0).expand(bsz, -1)

        xs = encoder.forward(
            x,
            mask=mask,
            sent_ids=sent_ids,
            type_ids=type_ids,
            pos_ids=pos_ids,
            stance_logit=stance_logit,
            disco_logit=disco_logit
        )[0]

        x1_split_sizes = [x1_indices[0]] + [x1_indices[i] - x1_indices[i-1] for i in range(1, len(x1_indices))]
        x2_split_sizes = [x2_indices[0]] + [x2_indices[i] - x2_indices[i-1] for i in range(1, len(x2_indices))]
        xs = th.split(xs, x1_split_sizes + x2_split_sizes, dim=1)
        masks = th.split(x1_mask, x1_split_sizes, dim=1) + th.split(x2_mask, x2_split_sizes, dim=1)

        return xs, masks

    def forward(
        self,
        x1,
        x2,
        x1_mask=None,
        x2_mask=None,
        x1_sent_ids=None,
        x2_sent_ids=None,
        stance_logit=None,
        disco_logit=None
    ):
        if isinstance(self, nn.DataParallel):
            encoder = self.module.encoder
            matching_layer = self.module.matching_layer
            fusion_layer = self.module.fusion_layer
            conv_layer = self.module.conv_layer
            pos_emb = self.module.pos_emb
            trans_layer = self.module.trans_layer
            gru_layer = self.module.gru_layer
            fc_layer = self.module.fc_layer
            drop = self.module.drop
        else:
            encoder = self.encoder
            matching_layer = self.matching_layer
            fusion_layer = self.fusion_layer
            conv_layer = self.conv_layer
            pos_emb = self.pos_emb
            trans_layer = self.trans_layer
            gru_layer = self.gru_layer
            fc_layer = self.fc_layer
            drop = self.drop

        bsz, x1_len = x1.size()
        x2_len = x2.size(1)

        if x1_mask is None:
            x1_mask = th.ones((bsz, x1_len), dtype=th.bool, device=x1.device)
        if x2_mask is None:
            x2_mask = th.ones((bsz, x2_len), dtype=th.bool, device=x2.device)

        if x1_sent_ids is not None:
            x1_indices = process_indices(x1_sent_ids).tolist()
        else:
            x1_sent_ids = th.zeros_like(x1)
            x1_indices = [x1_len]

        xs, masks = self.encode(
            x1,
            x2,
            x1_mask=x1_mask,
            x2_mask=x2_mask,
            x1_sent_ids=x1_sent_ids,
            x2_sent_ids=x2_sent_ids,
            stance_logit=stance_logit,
            disco_logit=disco_logit
        )

        if matching_layer is not None:
            zeros = th.zeros((bsz, 1, matching_layer.get_output_dim()), dtype=xs[0].dtype, device=xs[0].device)
            m_forwards = []
            m_backwords = []
            ms = []
            for i in range(1, len(xs)):
                if i == 1:
                    m_forwards.append(zeros.expand(-1, xs[0].size(1), -1))
                m1, m2 = matching_layer(xs[i - 1], xs[i], masks[i - 1], masks[i])
                m1, m2 = drop(th.cat(m1, dim=2)), drop(th.cat(m2, dim=2))
                m_backwords.append(m1)
                m_forwards.append(m2)
                if i == len(xs) - 1:
                    m_backwords.append(zeros.expand(-1, xs[-1].size(1), -1))
            for i in range(len(xs)):
                ms.append(th.cat([xs[i], m_forwards[i], m_backwords[i]], dim=-1))
            xs = tuple(ms)

        if fusion_layer is not None:
            fs = []
            for i in range(len(xs)):
                f = fusion_layer(xs[i], xs[i], xs[i], query_mask=masks[i], key_mask=masks[i])
                f = drop(f)
                fs.append(f)
            xs = tuple(fs)

        if conv_layer is not None:
            # x1_feat = conv_layer(th.cat(xs[:len(x1_indices)], dim=1))
            x2_feat = conv_layer(th.cat(xs[len(x1_indices):], dim=1))
            feat = x2_feat
        elif trans_layer is not None:
            rep_idx = encoder.get_special_rep_idx(x2)
            if rep_idx == x2_len-1:
                rep_idx = -1
            ts = []
            tsm = []
            for i in range(len(xs)):
                ts.append(xs[i][:, rep_idx])
                tsm.append(masks[i][:, rep_idx])
            ts = th.stack(ts, dim=1)
            tsm = th.stack(tsm, dim=1)
            if pos_emb is not None:
                pos_ids = th.arange(ts.size(1)-1, -1, -1, dtype=th.long, device=ts.device)
                ts = ts + pos_emb(pos_ids).unsqueeze(0)
            ts = trans_layer(ts, ts, ts, tsm, tsm)
            start, end = batch_convert_mask_to_start_and_end(tsm)
            if rep_idx == 0:
                # x1_feat = th.gather(
                #     ts,
                #     dim=1,
                #     index=start.unsqueeze(1).unsqueeze(1).expand(-1, -1, ts.size(-1))
                # ).squeeze(1)
                x2_feat = ts[:, len(x1_indices)]
            else:
                # x1_feat = ts[:, len(x1_indices)-1]
                x2_feat = th.gather(
                    ts,
                    dim=1,
                    index=end.unsqueeze(1).unsqueeze(1).expand(-1, -1, ts.size(-1))
                ).squeeze(1)
            feat = x2_feat
        elif gru_layer is not None:
            rep_idx = encoder.get_special_rep_idx(x2)
            if rep_idx == x2_len-1:
                rep_idx = -1
            g = []
            gm = []
            for i in range(len(xs)):
                g.append(xs[i][:, rep_idx])
                gm.append(masks[i][:, rep_idx])
            g = th.stack(g, dim=1)
            gm = th.stack(gm, dim=1)
            g = gru_layer(g)[0]
            start, end = batch_convert_mask_to_start_and_end(gm)
            if rep_idx == 0:
                # x1_feat = th.gather(
                #     g,
                #     dim=1,
                #     index=start.unsqueeze(1).unsqueeze(1).expand(-1, -1, g.size(-1))
                # ).squeeze(1)
                x2_feat = g[:, len(x1_indices)]
            else:
                # x1_feat = g[:, len(x1_indices)-1]
                x2_feat = th.gather(
                    g,
                    dim=1,
                    index=end.unsqueeze(1).unsqueeze(1).expand(-1, -1, g.size(-1))
                ).squeeze(1)
            feat = x2_feat

        feat = drop(feat)
        output = fc_layer(feat)

        return output  # unnormalized results


class IntervalModel(FlatModel):
    def __init__(self, **kw):
        super(IntervalModel, self).__init__(**kw)

    def encode(
        self,
        x1,
        x2,
        x1_mask=None,
        x2_mask=None,
        x1_sent_ids=None,
        x2_sent_ids=None,
        stance_logit=None,
        disco_logit=None
    ):
        if isinstance(self, nn.DataParallel):
            encoder = self.module.encoder
        else:
            encoder = self.encoder

        bsz, x1_len = x1.size()
        x2_len = x2.size(1)
        x_len = x1_len + x2_len

        if x1_mask is None:
            x1_mask = th.ones((bsz, x1_len), dtype=th.bool, device=x1.device)
        if x2_mask is None:
            x2_mask = th.ones((bsz, x2_len), dtype=th.bool, device=x2.device)

        if x1_sent_ids is not None:
            x1_indices = process_indices(x1_sent_ids)
        else:
            x1_sent_ids = th.zeros_like(x1)
            x1_indices = th.tensor([x1_len], dtype=th.long, device=x1.device)
        if x2_sent_ids is not None:
            x2_indices = process_indices(x2_sent_ids)
        else:
            x2_sent_ids = th.zeros_like(x2)
            x2_indices = th.tensor([x2_len], dtype=th.long, device=x2.device)

        x1_type_ids, x2_type_ids = process_type_ids(x1_indices, x2_indices, method="interval")
        x1_indices, x2_indices = x1_indices.tolist(), x2_indices.tolist()

        x = th.cat([x1, x2], dim=1)
        mask = th.cat([x1_mask, x2_mask], dim=1)
        # pos_ids = th.cumsum(mask, dim=1).masked_fill(th.logical_not(mask), 0)
        pos_ids = None
        sent_ids = th.cat([x1_sent_ids, x2_sent_ids+x1_sent_ids[:, -1:]+1], dim=1)
        type_ids = th.cat([x1_type_ids, x2_type_ids], dim=0).unsqueeze(0).expand(bsz, -1)

        xs = encoder.forward(
            x,
            mask=mask,
            sent_ids=sent_ids,
            type_ids=type_ids,
            pos_ids=pos_ids,
            stance_logit=stance_logit,
            disco_logit=disco_logit
        )[0]

        x1_split_sizes = [x1_indices[0]] + [x1_indices[i] - x1_indices[i-1] for i in range(1, len(x1_indices))]
        x2_split_sizes = [x2_indices[0]] + [x2_indices[i] - x2_indices[i-1] for i in range(1, len(x2_indices))]
        xs = th.split(xs, x1_split_sizes + x2_split_sizes, dim=1)
        masks = th.split(x1_mask, x1_split_sizes, dim=1) + th.split(x2_mask, x2_split_sizes, dim=1)

        return xs, masks


class SegmentedModel(FlatModel):
    def __init__(self, **kw):
        super(SegmentedModel, self).__init__(**kw)

    def encode(
        self,
        x1,
        x2,
        x1_mask=None,
        x2_mask=None,
        x1_sent_ids=None,
        x2_sent_ids=None,
        stance_logit=None,
        disco_logit=None
    ):
        if isinstance(self, nn.DataParallel):
            encoder = self.module.encoder
        else:
            encoder = self.encoder

        bsz, x1_len = x1.size()
        x2_len = x2.size(1)
        x_len = x1_len + x2_len

        if x1_mask is None:
            x1_mask = th.ones((bsz, x1_len), dtype=th.bool, device=x1.device)
        if x2_mask is None:
            x2_mask = th.ones((bsz, x2_len), dtype=th.bool, device=x2.device)

        if x1_sent_ids is not None:
            x1_indices = process_indices(x1_sent_ids)
        else:
            x1_sent_ids = th.zeros_like(x1)
            x1_indices = th.tensor([x1_len], dtype=th.long, device=x1.device)
        if x2_sent_ids is not None:
            x2_indices = process_indices(x2_sent_ids)
        else:
            x2_sent_ids = th.zeros_like(x2)
            x2_indices = th.tensor([x2_len], dtype=th.long, device=x2.device)

        x1_type_ids, x2_type_ids = process_type_ids(x1_indices, x2_indices, method="segmented")
        num_context = th.max(x1_type_ids, dim=0, keepdim=True)[0] + 1
        dummy_type_ids = self.max_num_context - num_context
        x1_type_ids = x1_type_ids + dummy_type_ids
        x2_type_ids = x2_type_ids + dummy_type_ids
        x1_indices, x2_indices = x1_indices.tolist(), x2_indices.tolist()

        x = th.cat([x1, x2], dim=1)
        mask = th.cat([x1_mask, x2_mask], dim=1)
        # pos_ids = th.cumsum(mask, dim=1).masked_fill(th.logical_not(mask), 0)
        pos_ids = None
        sent_ids = th.cat([x1_sent_ids, x2_sent_ids+x1_sent_ids[:, -1:]+1], dim=1)
        type_ids = th.cat([x1_type_ids, x2_type_ids], dim=0).unsqueeze(0).expand(bsz, -1)
        indices = x1_indices + [x_len]

        xs = []
        # cls_pad = th.empty((bsz, 1), dtype=x.dtype, device=x.device).fill_(encoder.get_special_token_id(CLS))
        for i in range(len(indices)):
            if i == 0:
                j = 0
                k = indices[0]
                mems = None
                mmk = None
            else:
                j = indices[i-1]
                k = indices[i]
                mmk = mask[:, j-mems[0].size(1):j]

            inp = x[:, j:k]
            mk = mask[:, j:k]
            sd = sent_ids[:, j:k]
            td = type_ids[:, j:k] - type_ids[:, j:(j+1)] # zero-one
            if i == len(indices) - 1:
                td = td + 2
            # pd = th.cumsum(mk, dim=1).masked_fill(th.logical_not(mk), 0)
            pd = None
            sl = stance_logit[:, j:k] if stance_logit is not None else None
            dl = disco_logit[:, j:k] if disco_logit is not None else None
            feat, mems = encoder.forward(
                inp,
                mask=mk,
                sent_ids=sd,
                type_ids=td,
                pos_ids=pd,
                mems=mems,
                mems_mask=mmk,
                stance_logit=sl,
                disco_logit=dl
            )
            xs.append(feat)

        x1_split_sizes = [x1_indices[0]] + [x1_indices[i] - x1_indices[i-1] for i in range(1, len(x1_indices))]
        x2_split_sizes = [x2_indices[0]] + [x2_indices[i] - x2_indices[i-1] for i in range(1, len(x2_indices))]
        xs = tuple(xs[:-1]) + th.split(xs[-1], x2_split_sizes, dim=1)
        masks = th.split(x1_mask, x1_split_sizes, dim=1) + th.split(x2_mask, x2_split_sizes, dim=1)

        return xs, masks


class ContextualizedModel(FlatModel):
    def __init__(self, **kw):
        assert kw.get("encoder", "roberta") != "lstm"
        super(ContextualizedModel, self).__init__(**kw)

    def encode(
        self,
        x1,
        x2,
        x1_mask=None,
        x2_mask=None,
        x1_sent_ids=None,
        x2_sent_ids=None,
        stance_logit=None,
        disco_logit=None
    ):
        if isinstance(self, nn.DataParallel):
            encoder = self.module.encoder
        else:
            encoder = self.encoder

        bsz, x1_len = x1.size()
        x2_len = x2.size(1)
        x_len = x1_len + x2_len

        if x1_mask is None:
            x1_mask = th.ones((bsz, x1_len), dtype=th.bool, device=x1.device)
        if x2_mask is None:
            x2_mask = th.ones((bsz, x2_len), dtype=th.bool, device=x2.device)

        if x1_sent_ids is not None:
            x1_indices = process_indices(x1_sent_ids)
        else:
            x1_sent_ids = th.zeros_like(x1)
            x1_indices = th.tensor([x1_len], dtype=th.long, device=x1.device)
        if x2_sent_ids is not None:
            x2_indices = process_indices(x2_sent_ids)
        else:
            x2_sent_ids = th.zeros_like(x2)
            x2_indices = th.tensor([x2_len], dtype=th.long, device=x2.device)

        x1_type_ids, x2_type_ids = process_type_ids(x1_indices, x2_indices, method="interval")
        x1_indices, x2_indices = x1_indices.tolist(), x2_indices.tolist()

        x = th.cat([x1, x2], dim=1)
        mask = th.cat([x1_mask, x2_mask], dim=1)
        # pos_ids = th.cumsum(mask, dim=1).masked_fill(th.logical_not(mask), 0)
        pos_ids = None
        sent_ids = th.cat([x1_sent_ids, x2_sent_ids+x1_sent_ids[:, -1:]+1], dim=1)
        type_ids = th.cat([x1_type_ids, x2_type_ids], dim=0).unsqueeze(0).expand(bsz, -1)
        indices = x1_indices + [x_len]

        context_mask = th.zeros((bsz, x_len, x_len), dtype=th.bool, device=x.device)
        for i in range(len(indices)):
            j = indices[i - 2] if i - 2 >= 0 else 0
            k = indices[i + 1] if i + 1 < len(indices) else indices[i]
            context_mask[:, (indices[i-1] if i - 1 >= 0 else 0):indices[i], j:k].data.copy_(mask[:, j:k].unsqueeze(1))
        xs = encoder.forward(
            x,
            mask=context_mask,
            sent_ids=sent_ids,
            type_ids=type_ids,
            pos_ids=pos_ids,
            stance_logit=stance_logit,
            disco_logit=disco_logit
        )[0]

        x1_split_sizes = [x1_indices[0]] + [x1_indices[i] - x1_indices[i-1] for i in range(1, len(x1_indices))]
        x2_split_sizes = [x2_indices[0]] + [x2_indices[i] - x2_indices[i-1] for i in range(1, len(x2_indices))]
        xs = th.split(xs, x1_split_sizes + x2_split_sizes, dim=1)
        masks = th.split(x1_mask, x1_split_sizes, dim=1) + th.split(x2_mask, x2_split_sizes, dim=1)

        return xs, masks


class ConcatCell(nn.Module):
    def __init__(self, input_dim):
        super(ConcatCell, self).__init__()
        self.input_dim = input_dim

    def forward(self, x1, x2):
        return th.cat([x1, x2], dim=-1)

    def get_output_dim(self):
        return self.input_dim * 2


class GRUCell(nn.Module):
    def __init__(self, input_dim):
        super(GRUCell, self).__init__()

        self.input_dim = input_dim
        self.r_net = nn.Linear(input_dim * 2, input_dim)
        self.z_net = nn.Linear(input_dim * 2, input_dim)
        self.o_net = nn.Linear(input_dim * 2, input_dim)

        # init
        init_weight(self.r_net, activation="sigmoid", init="uniform")
        init_weight(self.z_net, activation="sigmoid", init="uniform")
        init_weight(self.o_net, activation="tanh", init="uniform")

    def forward(self, x1, x2):
        x = th.cat([x1, x2], dim=-1)

        r = F.sigmoid(self.r_net(x))
        z = F.sigmoid(self.z_net(x))
        o = F.tanh(self.o_net(th.cat([x1, r * x2], dim=-1)))

        return (1 - z) * x1 + z * o

    def get_output_dim(self):
        return self.input_dim


class HighwayCell(nn.Module):
    def __init__(self, input_dim):
        super(HighwayCell, self).__init__()

        self.input_dim = input_dim
        self.highway = Highway(input_dim * 2, activation="tanh")
        self.z_net = nn.Linear(input_dim * 2, input_dim)

        # init
        init_weight(self.z_net, activation="sigmoid", init="uniform")

    def forward(self, x1, x2):
        size = x1.size()
        dim = x1.size(-1)
        x = th.cat([x1, x2], dim=-1)
        o = self.highway(x)
        x = x.view(-1, 2, dim)
        o = o.view(-1, 2, dim)
        z = self.z_net(th.cat([x * o, th.abs(x - o)], dim=2))
        z = F.softmax(z, dim=1)
        o = th.sum(z * o, dim=1)
        o = o.view(size)

        return o

    def get_output_dim(self):
        return self.input_dim


class AttnCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=1):
        super(AttnCell, self).__init__()

        self.input_dim = input_dim

        self.attn = DotAttention(
            query_dim=input_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            scale=1/input_dim**0.5,
            score_func="softmax",
            add_zero_attn=False,
            add_residual=False,
            add_gate=True,
            pre_lnorm=True,
            post_lnorm=False
        )

    def forward(self, x1, x2, mask=None):
        o = self.attn(x2, x1, x1, mask, mask)

        return o

    def get_output_dim(self):
        return self.input_dim


class DisCOCModel(FlatModel):
    def __init__(self, **kw):
        super(DisCOCModel, self).__init__(**kw)
        num_perspectives = kw.get("num_perspectives", 8)
        hidden_dim = kw.get("hidden_dim", 128)

        encoder_dim = self.encoder.get_output_dim()
        # self.cell = ConcatCell(encoder_dim)
        # self.cell = GRUCell(encoder_dim)
        # self.cell = HighwayCell(encoder_dim)
        self.cell = AttnCell(encoder_dim, hidden_dim, num_perspectives)

        if self.cell.get_output_dim() != encoder_dim:
            if self.matching_layer is not None:
                del self.matching_layer
            if self.fusion_layer is not None:
                del self.fusion_layer
            if self.conv_layer is not None:
                del self.conv_layer
            if self.trans_layer is not None:
                del self.trans_layer
            self.create_layers(self.encoder.get_output_dim())

    def encode(
        self,
        x1,
        x2,
        x1_mask=None,
        x2_mask=None,
        x1_sent_ids=None,
        x2_sent_ids=None,
        stance_logit=None,
        disco_logit=None
    ):
        if isinstance(self, nn.DataParallel):
            encoder = self.module.encoder
            cell = self.module.cell
        else:
            encoder = self.encoder
            cell = self.cell

        bsz, x1_len = x1.size()
        x2_len = x2.size(1)
        x_len = x1_len + x2_len

        if x1_mask is None:
            x1_mask = th.ones((bsz, x1_len), dtype=th.bool, device=x1.device)
        if x2_mask is None:
            x2_mask = th.ones((bsz, x2_len), dtype=th.bool, device=x2.device)

        if x1_sent_ids is not None:
            x1_indices = process_indices(x1_sent_ids)
        else:
            x1_sent_ids = th.zeros_like(x1)
            x1_indices = th.tensor([x1_len], dtype=th.long, device=x1.device)
        if x2_sent_ids is not None:
            x2_indices = process_indices(x2_sent_ids)
        else:
            x2_sent_ids = th.zeros_like(x2)
            x2_indices = th.tensor([x2_len], dtype=th.long, device=x2.device)

        x1_type_ids, x2_type_ids = process_type_ids(x1_indices, x2_indices, method="segmented")
        num_context = th.max(x1_type_ids, dim=0, keepdim=True)[0] + 1
        dummy_type_ids = self.max_num_context - num_context
        x1_type_ids = x1_type_ids + dummy_type_ids
        x2_type_ids = x2_type_ids + dummy_type_ids
        x1_indices, x2_indices = x1_indices.tolist(), x2_indices.tolist()

        x = th.cat([x1, x2], dim=1)
        mask = th.cat([x1_mask, x2_mask], dim=1)
        sent_ids = th.cat([x1_sent_ids, x2_sent_ids+x1_sent_ids[:, -1:]+1], dim=1)
        type_ids = th.cat([x1_type_ids, x2_type_ids], dim=0).unsqueeze(0).expand(bsz, -1)
        indices = x1_indices + [x_len]

        x_forwards = []
        x_backwards = []
        for i in range(len(indices)):
            if i == 0:
                j = 0
                k = indices[0]
                sd = th.zeros((bsz, k), dtype=th.long, device=x.device)
                td = th.ones((bsz, k), dtype=th.long, device=x.device)
            else:
                if i == 1:
                    j = 0
                else:
                    j = indices[i - 2]
                k = indices[i]
                sd = sent_ids[:, j:k]
                td = type_ids[:, j:k] - type_ids[:, j:(j+1)] # zero-one
            inp = x[:, j:k]
            mk = mask[:, j:k]
            # pd = th.cumsum(mk, dim=1).masked_fill(th.logical_not(mk), 0)
            pd = None

            if stance_logit is None or i == 0:
                sl = None
            else:
                dummy_sl = th.zeros(
                    (bsz, indices[i-1]-j, stance_logit.size(-1)),
                    device=stance_logit.device,
                    dtype=stance_logit.dtype
                )
                dummy_sl[:, :, 0].fill_(INF)
                sl = th.cat([dummy_sl, stance_logit[:, indices[i-1]:k]], dim=1)
                # sl = stance_logit[:, j:k]

            if disco_logit is None or i == 0:
                dl = None
            else:
                dummy_dl = th.zeros(
                    (bsz, indices[i-1]-j, disco_logit.size(-1)),
                    device=disco_logit.device,
                    dtype=disco_logit.dtype
                )
                dummy_dl[:, :, 0].fill_(INF)
                dl = th.cat([dummy_dl, disco_logit[:, indices[i-1]:k]], dim=1)
                # dl = disco_logit[:, j:k]

            feat = encoder.forward(
                inp,
                mask=mk,
                sent_ids=sd,
                type_ids=td,
                pos_ids=pd,
                stance_logit=sl,
                disco_logit=dl
            )[0]

            if i == 0:
                x_forwards.append(feat)
            else:
                x_backwards.append(feat[:, :indices[i-1]-j])
                x_forwards.append(feat[:, indices[i-1]-j:])

            if i == len(indices) - 1:  # text
                j = indices[i-1] if i > 0 else 0
                k = indices[i]
                inp = x[:, j:k]
                mk = mask[:, j:k]
                sd = sent_ids[:, j:k]
                td = type_ids[:, j:k] - type_ids[:, j:(j+1)] + 2
                # pd = th.cumsum(mk, dim=1).masked_fill(th.logical_not(mk), 0)
                pd = None

                if stance_logit is None:
                    sl = None
                else:
                    dummy_sl = th.zeros(
                        (bsz, indices[i-1]-j, stance_logit.size(-1)),
                        device=stance_logit.device,
                        dtype=stance_logit.dtype
                    )
                    dummy_sl[:, :, 0].fill_(INF)
                    sl = th.cat([dummy_sl, stance_logit[:, indices[i-1]:k]], dim=1)
                    # sl = stance_logit[:, j:k]

                if disco_logit is None:
                    dl = None
                else:
                    dummy_dl = th.zeros(
                        (bsz, indices[i-1]-j, disco_logit.size(-1)),
                        device=disco_logit.device,
                        dtype=disco_logit.dtype
                    )
                    dummy_dl[:, :, 0].fill_(INF)
                    dl = th.cat([dummy_dl, disco_logit[:, indices[i-1]:k]], dim=1)
                    # dl = disco_logit[:, j:k]

                feat = encoder.forward(
                    inp,
                    mask=mk,
                    sent_ids=sd,
                    type_ids=td,
                    pos_ids=pd,
                    stance_logit=sl,
                    disco_logit=dl
                )[0]

                x_backwards.append(feat)

        xs = []
        if isinstance(cell, AttnCell):
            for i in range(len(indices)):
                j = indices[i-1] if i > 0 else 0
                k = indices[i]
                xs.append(cell(x_forwards[i], x_backwards[i], mask[:, j:k]))
        else:
            for i in range(len(indices)):
                xs.append(cell(x_forwards[i], x_backwards[i]))

        x1_split_sizes = [x1_indices[0]] + [x1_indices[i] - x1_indices[i-1] for i in range(1, len(x1_indices))]
        x2_split_sizes = [x2_indices[0]] + [x2_indices[i] - x2_indices[i-1] for i in range(1, len(x2_indices))]
        xs = tuple(xs[:-1]) + th.split(xs[-1], x2_split_sizes, dim=1)
        masks = th.split(x1_mask, x1_split_sizes, dim=1) + th.split(x2_mask, x2_split_sizes, dim=1)

        return xs, masks

    # def encode(
    #     self,
    #     x1,
    #     x2,
    #     x1_mask=None,
    #     x2_mask=None,
    #     x1_sent_ids=None,
    #     x2_sent_ids=None,
    #     stance_logit=None,
    #     disco_logit=None
    # ):
    #     if isinstance(self, nn.DataParallel):
    #         encoder = self.module.encoder
    #         cell = self.module.cell
    #     else:
    #         encoder = self.encoder
    #         cell = self.cell

    #     bsz, x1_len = x1.size()
    #     x2_len = x2.size(1)
    #     x_len = x1_len + x2_len

    #     if x1_mask is None:
    #         x1_mask = th.ones((bsz, x1_len), dtype=th.bool, device=x1.device)
    #     if x2_mask is None:
    #         x2_mask = th.ones((bsz, x2_len), dtype=th.bool, device=x2.device)

    #     if x1_sent_ids is not None:
    #         x1_indices = process_indices(x1_sent_ids)
    #     else:
    #         x1_sent_ids = th.zeros_like(x1)
    #         x1_indices = th.tensor([x1_len], dtype=th.long, device=x1.device)
    #     if x2_sent_ids is not None:
    #         x2_indices = process_indices(x2_sent_ids)
    #     else:
    #         x2_sent_ids = th.zeros_like(x2)
    #         x2_indices = th.tensor([x2_len], dtype=th.long, device=x2.device)

    #     x1_type_ids, x2_type_ids = process_type_ids(x1_indices, x2_indices, method="segmented")
    #     num_context = th.max(x1_type_ids, dim=0, keepdim=True)[0] + 1
    #     dummy_type_ids = self.max_num_context - num_context
    #     x1_type_ids = x1_type_ids + dummy_type_ids
    #     x2_type_ids = x2_type_ids + dummy_type_ids
    #     x1_indices, x2_indices = x1_indices.tolist(), x2_indices.tolist()

    #     x = th.cat([x1, x2], dim=1)
    #     mask = th.cat([x1_mask, x2_mask], dim=1)
    #     # pos_ids = th.cumsum(mask, dim=1).masked_fill(th.logical_not(mask), 0)
    #     pos_ids = None
    #     sent_ids = th.cat([x1_sent_ids, x2_sent_ids+x1_sent_ids[:, -1:]+1], dim=1)
    #     type_ids = th.cat([x1_type_ids, x2_type_ids], dim=0).unsqueeze(0).expand(bsz, -1)
    #     indices = x1_indices + [x_len]

    #     dummy_ids = th.ones_like(sent_ids)
    #     clamped_ids = th.clamp(sent_ids, max=len(x1_indices)) # regard all x2 as a whole
    #     even_sent_ids = th.bitwise_or(clamped_ids, dummy_ids)
    #     even_mask = even_sent_ids.unsqueeze(1) == even_sent_ids.unsqueeze(2)
    #     odd_sent_ids = th.bitwise_or(clamped_ids + 1, dummy_ids)
    #     odd_mask = odd_sent_ids.unsqueeze(1) == odd_sent_ids.unsqueeze(2)
    #     even_mask.masked_fill_((mask == 0).unsqueeze(-1), 0)
    #     odd_mask.masked_fill_((mask == 0).unsqueeze(-1), 0)
    #     if len(indices) % 2 == 1:
    #         even_type_ids = th.cat(
    #             [
    #                 type_ids[:, :x1_len] % 2,
    #                 type_ids[:, x1_len:] - type_ids[:, x1_len:(x1_len+1)] + 2
    #             ],
    #             dim=1
    #         )
    #         odd_type_ids = (type_ids + 1) % 2
    #     else:
    #         even_type_ids = type_ids % 2
    #         odd_type_ids = th.cat(
    #             [
    #                 (type_ids[:, :x1_len] + 1) % 2,
    #                 type_ids[:, x1_len:] - type_ids[:, x1_len:(x1_len+1)] + 2
    #             ],
    #             dim=1
    #         )

    #     even_x = encoder.forward(
    #         x,
    #         mask=even_mask,
    #         sent_ids=sent_ids,
    #         type_ids=even_type_ids,
    #         pos_ids=pos_ids,
    #         stance_logit=stance_logit,
    #         disco_logit=disco_logit
    #     )[0]
    #     odd_x = encoder.forward(
    #         x,
    #         mask=odd_mask,
    #         sent_ids=sent_ids,
    #         type_ids=odd_type_ids,
    #         pos_ids=pos_ids,
    #         stance_logit=stance_logit,
    #         disco_logit=disco_logit
    #     )[0]

    #     x1_split_sizes = [x1_indices[0]] + [x1_indices[i] - x1_indices[i-1] for i in range(1, len(x1_indices))]
    #     x2_split_sizes = [x2_indices[0]] + [x2_indices[i] - x2_indices[i-1] for i in range(1, len(x2_indices))]
    #     even_xs = th.split(
    #         even_x,
    #         x1_split_sizes + x2_split_sizes,
    #         dim=1
    #     )
    #     odd_xs = th.split(
    #         odd_x,
    #         x1_split_sizes + x2_split_sizes,
    #         dim=1
    #     )
    #     masks = th.split(x1_mask, x1_split_sizes, dim=1) + th.split(x2_mask, x2_split_sizes, dim=1)

    #     xs = []
    #     if isinstance(cell, AttnCell):
    #         for i in range(len(even_xs)):
    #             if i % 2 == 0:
    #                 xs.append(cell(odd_xs[i], even_xs[i], masks[i]))
    #             else:
    #                 xs.append(cell(even_xs[i], odd_xs[i]))
    #     else:
    #         for i in range(len(even_xs)):
    #             if i % 2 == 0:
    #                 xs.append(cell(odd_xs[i], even_xs[i]))
    #             else:
    #                 xs.append(cell(even_xs[i], odd_xs[i]))

    #     return xs, masks


class HAN(nn.Module):
    def __init__(self, **kw):
        super(HAN, self).__init__()
        max_num_text = kw.get("max_num_text", 1)
        max_num_context = kw.get("max_num_context", 1)
        encoder = kw.get("encoder", "roberta")
        hidden_dim = kw.get("hidden_dim", 128)
        num_perspectives = kw.get("num_perspectives", 8)
        num_labels = kw.get("num_labels", 3)
        dropout = kw.get("dropout", 0.0)

        self.max_num_context = max_num_context
        self.max_num_text = max_num_text
        self.drop = nn.Dropout(dropout, inplace=False)
        if encoder == "bert":
            self.encoder = BertEncoder(num_segments=max_num_text+max_num_context+2, **kw)
            dim = self.encoder.get_output_dim()
            self.word_linear = nn.Linear(dim, dim)
            self.word_attn_vec = nn.Parameter(th.Tensor(dim))
            self.sent_encoder = TransformerLayer(
                input_dim=dim,
                hidden_dim=hidden_dim,
                num_heads=num_perspectives,
                add_residual=True,
                add_gate=False,
                pre_lnorm=True,
                post_lnorm=False,
                dropout=0.0
            )
            self.sent_linear = nn.Linear(dim, dim)
            self.sent_attn_vec = nn.Parameter(th.Tensor(dim))
        elif encoder == "albert":
            self.encoder = AlbertEncoder(num_segments=max_num_text+max_num_context+2, **kw)
            dim = self.encoder.get_output_dim()
            self.word_linear = nn.Linear(dim, dim)
            self.word_attn_vec = nn.Parameter(th.Tensor(dim))
            self.sent_encoder = TransformerLayer(
                input_dim=dim,
                hidden_dim=hidden_dim,
                num_heads=num_perspectives,
                add_residual=True,
                add_gate=False,
                pre_lnorm=True,
                post_lnorm=False,
                dropout=0.0
            )
            self.sent_linear = nn.Linear(dim, dim)
            self.sent_attn_vec = nn.Parameter(th.Tensor(dim))
        elif encoder == "roberta":
            self.encoder = RobertaEncoder(num_segments=max_num_text+max_num_context+2, **kw)
            dim = self.encoder.get_output_dim()
            self.word_linear = nn.Linear(dim, dim)
            self.word_attn_vec = nn.Parameter(th.Tensor(dim))
            self.sent_encoder = TransformerLayer(
                input_dim=dim,
                hidden_dim=hidden_dim,
                num_heads=num_perspectives,
                add_residual=True,
                add_gate=False,
                pre_lnorm=True,
                post_lnorm=False,
                dropout=0.0
            )
            self.sent_linear = nn.Linear(dim, dim)
            self.sent_attn_vec = nn.Parameter(th.Tensor(dim))
        elif encoder == "xlnet":
            self.encoder = XLNetEncoder(num_segments=max_num_text+max_num_context+2, **kw)
            dim = self.encoder.get_output_dim()
            self.word_linear = nn.Linear(dim, dim)
            self.word_attn_vec = nn.Parameter(th.Tensor(dim))
            self.sent_encoder = TransformerLayer(
                input_dim=dim,
                hidden_dim=hidden_dim,
                num_heads=num_perspectives,
                add_residual=True,
                add_gate=False,
                pre_lnorm=True,
                post_lnorm=False,
                dropout=0.0
            )
            self.sent_linear = nn.Linear(dim, dim)
            self.sent_attn_vec = nn.Parameter(th.Tensor(dim))
        elif encoder == "lstm":
            self.encoder = LSTMEncoder(num_segments=max_num_text+max_num_context+2, **kw)
            dim = self.encoder.get_output_dim()
            self.word_linear = nn.Linear(dim, dim)
            self.word_attn_vec = nn.Parameter(th.Tensor(dim))
            self.sent_encoder = nn.LSTM(
                input_size=dim,
                hidden_size=dim//2,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )
            self.sent_linear = nn.Linear(dim, dim)
            self.sent_attn_vec = nn.Parameter(th.Tensor(dim))
        else:
            raise NotImplementedError("Error: encoder=%s is not supported now." % (encoder))

        self.fc_layer = MLP(
            input_dim=dim,
            hidden_dim=hidden_dim,
            output_dim=num_labels,
            num_mlp_layers=2,
            activation="none",
            norm_layer="batch_norm"
        )
        self.drop = nn.Dropout(dropout)

        # init
        init_weight(self.word_attn_vec, init="uniform")
        init_weight(self.word_linear, init="uniform")
        init_weight(self.sent_attn_vec, init="uniform")
        init_weight(self.sent_linear, init="uniform")

    def set_finetune(self, finetune):
        assert finetune in ["full", "layers", "last", "type", "none"]

        for param in self.parameters():
            param.requires_grad = True

        self.encoder.set_finetune(finetune)

    def forward(
        self,
        x1,
        x2,
        x1_mask=None,
        x2_mask=None,
        x1_sent_ids=None,
        x2_sent_ids=None,
        stance_logit=None,
        disco_logit=None
    ):
        if isinstance(self, nn.DataParallel):
            encoder = self.module.encoder
            encoder = self.module.encoder
            word_attn_vec = self.module.word_attn_vec
            word_linear = self.module.word_linear
            sent_encoder = self.module.sent_encoder
            sent_attn_vec = self.module.sent_attn_vec
            sent_linear = self.module.sent_linear
            fc_layer = self.module.fc_layer
            drop = self.module.drop
        else:
            encoder = self.encoder
            encoder = self.encoder
            word_attn_vec = self.word_attn_vec
            word_linear = self.word_linear
            sent_encoder = self.sent_encoder
            sent_attn_vec = self.sent_attn_vec
            sent_linear = self.sent_linear
            fc_layer = self.fc_layer
            drop = self.drop

        bsz, x1_len = x1.size()
        x2_len = x2.size(1)
        x_len = x1_len + x2_len

        if x1_mask is None:
            x1_mask = th.ones((bsz, x1_len), dtype=th.bool, device=x1.device)
        if x2_mask is None:
            x2_mask = th.ones((bsz, x2_len), dtype=th.bool, device=x2.device)

        if x1_sent_ids is not None:
            x1_indices = process_indices(x1_sent_ids)
        else:
            x1_sent_ids = th.zeros_like(x1)
            x1_indices = th.tensor([x1_len], dtype=th.long, device=x1.device)
        if x2_sent_ids is not None:
            x2_indices = process_indices(x2_sent_ids)
        else:
            x2_sent_ids = th.zeros_like(x2)
            x2_indices = th.tensor([x2_len], dtype=th.long, device=x2.device)

        x1_type_ids, x2_type_ids = process_type_ids(x1_indices, x2_indices, method="segmented")
        num_context = th.max(x1_type_ids, dim=0, keepdim=True)[0] + 1
        dummy_type_ids = self.max_num_context - num_context
        x1_type_ids = x1_type_ids + dummy_type_ids
        x2_type_ids = x2_type_ids + dummy_type_ids
        x1_indices, x2_indices = x1_indices.tolist(), x2_indices.tolist()

        x = th.cat([x1, x2], dim=1)
        mask = th.cat([x1_mask, x2_mask], dim=1)
        sent_ids = th.cat([x1_sent_ids, x2_sent_ids+x1_sent_ids[:, -1:]+1], dim=1)
        type_ids = th.cat([x1_type_ids, x2_type_ids], dim=0).unsqueeze(0).expand(bsz, -1)
        indices = x1_indices + [x_len]

        sent_feats = []
        for i in range(len(indices)):
            if i == 0:
                j = 0
                k = indices[0]
            else:
                j = indices[i-1]
                k = indices[i]
            sd = th.zeros((bsz, k-j), dtype=th.long, device=x.device)
            td = th.zeros((bsz, k-j), dtype=th.long, device=x.device)
            inp = x[:, j:k]
            mk = mask[:, j:k]
            # pd = th.cumsum(mk, dim=1).masked_fill(th.logical_not(mk), 0)
            pd = None

            if stance_logit is None or i == 0:
                sl = None
            else:
                sl = stance_logit[:, j:k]

            if disco_logit is None or i == 0:
                dl = None
            else:
                dl = disco_logit[:, j:k]

            word_feat = encoder.forward(
                inp,
                mask=mk,
                sent_ids=sd,
                type_ids=td,
                pos_ids=pd,
                stance_logit=sl,
                disco_logit=dl
            )[0]

            attn_score = th.einsum(
                "bid,bjd->bij",
                (
                    word_linear(word_feat),
                    word_attn_vec.view(1, 1, -1).expand(bsz, -1, -1)
                )
            )
            attn_score.masked_fill_((mk == 0).unsqueeze(-1), _INF)
            attn_score = F.softmax(attn_score, dim=1)

            sent_feats.append(th.sum(word_feat * attn_score, dim=1))

        sent_feat = th.stack(sent_feats, dim=1)
        sent_feat = sent_encoder(sent_feat)
        if isinstance(sent_feat, tuple):
            sent_feat = sent_feat[0]
        attn_score = th.einsum(
            "bid,bjd->bij",
            (
                sent_linear(sent_feat),
                sent_attn_vec.view(1, 1, -1).expand(bsz, -1, -1)
            )
        )
        attn_score = F.softmax(attn_score, dim=1)

        feat = th.sum(sent_feat * attn_score, dim=1)

        feat = drop(feat)
        output = fc_layer(feat)

        return output