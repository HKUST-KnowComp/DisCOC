import torch as th
import torch.nn as nn
import torch.nn.functional as F
from container import Parallel, Sequential
from function import map_activation_str_to_layer
from function import masked_max, masked_mean, masked_softmax
from function import multi_perspective_match, multi_perspective_match_pairwise
from util import batch_convert_mask_to_start_and_end, init_weight
from util import init_weight

_INF = -1e30
EPS = 1e-8

class PositionEmbedding(nn.Module):
    def __init__(self, input_dim, max_len=512, scale=1):
        super(PositionEmbedding, self).__init__()

        freq_seq = th.arange(0, input_dim, 2.0, dtype=th.float)
        inv_freq = 1 / th.pow(10000, (freq_seq / input_dim))
        sinusoid_inp = th.ger(th.arange(0, max_len, 1.0), inv_freq)
        self.register_buffer("emb", th.cat([th.sin(sinusoid_inp), th.cos(sinusoid_inp)], dim=-1) * scale)

    def forward(self, x):
        size = x.size()
        emb = th.index_select(
            self.emb,
            dim=0,
            index=x.flatten()
        ).view(list(size) + [self.emb.size(-1)])

        return emb

    def get_output_dim(self):
        return self.emb.size(-1)


class DotAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        key_dim,
        value_dim,
        hidden_dim,
        num_heads=1,
        scale=1,
        score_func="softmax",
        add_zero_attn=False,
        add_gate=False,
        add_residual=False,
        pre_lnorm=False,
        post_lnorm=False,
        dropout=0.0
    ):
        super(DotAttention, self).__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scale = scale
        self.add_zero_attn = add_zero_attn
        self.add_residual = add_residual
        self.pre_lnorm = pre_lnorm
        self.post_lnorm = post_lnorm
        self.drop = nn.Dropout(dropout)

        if hidden_dim == -1 and \
           (query_dim != key_dim or query_dim != value_dim or key_dim != value_dim):
            raise ValueError(
                "Error: when hidden_dim equals 1, we need the query, key, and value have the same dimension!"
            )

        if hidden_dim != -1:
            self.weight_q = nn.Parameter(th.Tensor(query_dim, hidden_dim))
            self.weight_k = nn.Parameter(th.Tensor(key_dim, hidden_dim))
            self.weight_v = nn.Parameter(th.Tensor(value_dim, hidden_dim))
            self.weight_o = nn.Parameter(th.Tensor(hidden_dim, query_dim))
        else:
            self.register_parameter("weight_q", None)
            self.register_parameter("weight_k", None)
            self.register_parameter("weight_v", None)
            self.register_parameter("weight_o", None)

        self.score_act = map_activation_str_to_layer(score_func)
        if hasattr(self.score_act, "dim"):
            setattr(self.score_act, "dim", 2)  # not the last dimension
        if add_gate:
            self.g_net = nn.Linear(query_dim * 2, query_dim, bias=True)
        else:
            self.g_net = None
        if pre_lnorm:
            self.q_layer_norm = nn.LayerNorm(query_dim)
            self.k_layer_norm = nn.LayerNorm(key_dim)
            self.v_layer_norm = nn.LayerNorm(value_dim)
            self.o_layer_norm = None
        if post_lnorm:
            self.q_layer_norm = None
            self.k_layer_norm = None
            self.v_layer_norm = None
            self.o_layer_norm = nn.LayerNorm(query_dim)

        # init
        if hidden_dim != -1:
            init_weight(self.weight_q, init="uniform")
            init_weight(self.weight_k, init="uniform")
            init_weight(self.weight_v, init="uniform")
            init_weight(self.weight_o, init="uniform")
        if pre_lnorm:
            init_weight(self.q_layer_norm)
            init_weight(self.k_layer_norm)
            init_weight(self.v_layer_norm)
        if post_lnorm:
            init_weight(self.o_layer_norm)

    def compute_score(self, query, key, key_mask=None):
        bsz = query.size(0)
        qlen, klen = query.size(1), key.size(1)

        if self.weight_q is not None:
            query = th.matmul(query, self.weight_q)
            key = th.matmul(key, self.weight_k)

        query = query.view(bsz, qlen, self.num_heads, -1)
        key = key.view(bsz, klen, self.num_heads, -1)

        # [bsz x qlen x klen x num_heads]
        attn_score = th.einsum("bind,bjnd->bijn", (query, key))
        attn_score.mul_(self.scale)

        if key_mask is not None:
            if key_mask.dim() < attn_score.dim():
                key_mask = key_mask.unsqueeze(-1)
            while key_mask.dim() < attn_score.dim():
                key_mask = key_mask.unsqueeze(1)
            attn_score.masked_fill_(key_mask == 0, _INF)

        # [bsz x qlen x klen x num_heads]
        if self.score_act is not None:
            attn_score = self.score_act(attn_score)

        return attn_score

    def forward(self, query, key, value, query_mask=None, key_mask=None):
        bsz = query.size(0)
        qlen, klen, vlen = query.size(1), key.size(1), value.size(1)

        original_query = query

        if self.add_zero_attn:
            key = th.cat([key, th.zeros((bsz, 1) + key.size()[2:], dtype=key.dtype, device=key.device)], dim=1)
            value = th.cat(
                [
                    value,
                    th.zeros((bsz, 1) + value.size()[2:], dtype=value.dtype, device=value.device)
                ],
                dim=1
            )
            if key_mask is not None:
                key_mask = th.cat(
                    [
                        key_mask, th.ones((bsz, 1), dtype=key_mask.dtype, device=key_mask.device)
                    ],
                    dim=1
                )

        if self.pre_lnorm:
            # layer normalization
            query = self.q_layer_norm(query)
            key = self.k_layer_norm(key)
            value = self.v_layer_norm(value)

        attn_score = self.compute_score(query, key, key_mask)
        attn_score = self.drop(attn_score)

        # [bsz x qlen x klen x num_heads] x [bsz x klen x num_heads x head_dim] -> [bsz x qlen x num_heads x head_dim]
        if self.weight_v is not None:
            value = th.matmul(value, self.weight_v)
        value = value.view(bsz, vlen, self.num_heads, -1)
        attn_vec = th.einsum("bijn,bjnd->bind", (attn_score, value))
        attn_vec = attn_vec.contiguous().view(bsz, qlen, -1)

        if query_mask is not None:
            while query_mask.dim() < attn_vec.dim():
                query_mask = query_mask.unsqueeze(-1)
            attn_vec.masked_fill_(query_mask == 0, 0)

        if self.weight_o is not None:
            attn_vec = th.matmul(attn_vec, self.weight_o)
        attn_vec = self.drop(attn_vec)

        if self.g_net is not None:
            g = F.sigmoid(self.g_net(th.cat([original_query, attn_vec], dim=-1)))
            attn_out = g * original_query + (1 - g) * attn_vec
        else:
            attn_out = attn_vec

        if self.add_residual:
            attn_out = original_query + attn_out

        if self.post_lnorm:
            attn_out = self.o_layer_norm(attn_out)

        return attn_out

    def get_output_dim(self):
        return self.query_dim


class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=1,
        add_residual=True,
        add_gate=False,
        pre_lnorm=True,
        post_lnorm=False,
        dropout=0.0
    ):
        super(TransformerLayer, self).__init__()

        self.cross_attn = DotAttention(
            query_dim=input_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            scale=1 / hidden_dim**0.5,
            score_func="softmax",
            add_zero_attn=False,
            add_residual=add_residual,
            add_gate=add_gate,
            pre_lnorm=pre_lnorm,
            post_lnorm=post_lnorm,
            dropout=dropout
        )
        self.self_attn = DotAttention(
            query_dim=input_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            scale=1 / hidden_dim**0.5,
            score_func="softmax",
            add_zero_attn=False,
            add_residual=add_residual,
            add_gate=add_gate,
            pre_lnorm=pre_lnorm,
            post_lnorm=post_lnorm,
            dropout=dropout
        )
        self.linear = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

        # init
        init_weight(self.linear, init="uniform")
        init_weight(self.layer_norm)

    def forward(self, query, key=None, value=None, query_mask=None, key_mask=None):
        if key is None:
            key = query
        if value is None:
            value = query
        x = self.cross_attn(query, key, value, query_mask=query_mask, key_mask=key_mask)
        x = self.self_attn(x, x, x, query_mask=query_mask, key_mask=query_mask)
        x = self.layer_norm(x + self.linear(x))

        return x


class BiMpmMatching(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_perspectives,
        share_weights_between_directions=True,
        with_full_match=True,
        with_maxpool_match=True,
        with_attentive_match=True,
        with_max_attentive_match=True
    ):
        super(BiMpmMatching, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_perspectives = num_perspectives

        self.with_full_match = with_full_match
        self.with_maxpool_match = with_maxpool_match
        self.with_attentive_match = with_attentive_match
        self.with_max_attentive_match = with_max_attentive_match

        if not (with_full_match or with_maxpool_match or with_attentive_match or with_max_attentive_match):
            raise ValueError("At least one of the matching method should be enabled")

        def create_parameter():  # utility function to create and initialize a parameter
            param = nn.Parameter(th.Tensor(num_perspectives, hidden_dim))
            init_weight(param)
            return param

        self.output_dim = 2  # used to calculate total output dimension, 2 is for cosine max and cosine min
        weights = []
        if with_full_match:
            self.full_forward_match_weights = nn.Parameter(th.Tensor(num_perspectives, hidden_dim))
            if share_weights_between_directions:
                self.full_forward_match_weights_reversed = self.full_forward_match_weights
            else:
                self.full_forward_match_weights_reversed = nn.Parameter(th.Tensor(num_perspectives, hidden_dim))
            self.full_backward_match_weights = nn.Parameter(th.Tensor(num_perspectives, hidden_dim))
            if share_weights_between_directions:
                self.full_backward_match_weights_reversed = self.full_backward_match_weights
            else:
                self.full_backward_match_weights_reversed = nn.Parameter(th.Tensor(num_perspectives, hidden_dim))
            self.output_dim += (num_perspectives + 1) * 2
            weights.append(self.full_forward_match_weights_reversed)
            weights.append(self.full_backward_match_weights)

        if with_maxpool_match:
            self.maxpool_match_weights = nn.Parameter(th.Tensor(num_perspectives, hidden_dim))
            self.output_dim += num_perspectives * 2
            weights.append(self.maxpool_match_weights)

        if with_attentive_match:
            self.attentive_match_weights = nn.Parameter(th.Tensor(num_perspectives, hidden_dim))
            if share_weights_between_directions:
                self.attentive_match_weights_reversed = self.attentive_match_weights
            else:
                self.attentive_match_weights_reversed = nn.Parameter(th.Tensor(num_perspectives, hidden_dim))
            self.output_dim += num_perspectives + 1
            weights.append(self.attentive_match_weights)

        if with_max_attentive_match:
            self.max_attentive_match_weights = nn.Parameter(th.Tensor(num_perspectives, hidden_dim))
            if share_weights_between_directions:
                self.max_attentive_match_weights_reversed = self.max_attentive_match_weights
            else:
                self.max_attentive_match_weights_reversed = nn.Parameter(th.Tensor(num_perspectives, hidden_dim))
            self.output_dim += num_perspectives + 1
            weights.append(self.max_attentive_match_weights)

        # init
        for weight in weights:
            init_weight(weight)

    def forward(self, context_1, context_2, mask_1, mask_2):
        assert (not mask_2.requires_grad) and (not mask_1.requires_grad)
        assert context_1.size(-1) == context_2.size(-1) == self.hidden_dim

        # explicitly set masked weights to zero
        # (batch_size, seq_len*, hidden_dim)
        context_1 = context_1.masked_fill((mask_1 == 0).unsqueeze(-1), 0.0)
        context_2 = context_2.masked_fill((mask_2 == 0).unsqueeze(-1), 0.0)

        # array to keep the matching vectors for the two sentences
        matching_vector_1 = []
        matching_vector_2 = []

        # Step 0. unweighted cosine
        # First calculate the cosine similarities between each forward
        # (or backward) contextual embedding and every forward (or backward)
        # contextual embedding of the other sentence.

        # (batch, seq_len1, seq_len2)
        cosine_sim = F.cosine_similarity(context_1.unsqueeze(-2), context_2.unsqueeze(-3), dim=3)

        # (batch, seq_len*, 1)
        cosine_max_1 = masked_max(cosine_sim, mask_2.unsqueeze(-2), dim=2, keepdim=True)
        cosine_mean_1 = masked_mean(cosine_sim, mask_2.unsqueeze(-2), dim=2, keepdim=True)
        cosine_max_2 = masked_max(cosine_sim.permute(0, 2, 1), mask_1.unsqueeze(-2), dim=2, keepdim=True)
        cosine_mean_2 = masked_mean(cosine_sim.permute(0, 2, 1), mask_1.unsqueeze(-2), dim=2, keepdim=True)

        matching_vector_1.extend([cosine_max_1, cosine_mean_1])
        matching_vector_2.extend([cosine_max_2, cosine_mean_2])

        # Step 1. Full-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with the last time step of the forward (or backward)
        # contextual embedding of the other sentence
        if self.with_full_match:
            # (batch, 1, hidden_dim)

            start_1, end_1 = batch_convert_mask_to_start_and_end(mask_1)
            start_2, end_2 = batch_convert_mask_to_start_and_end(mask_2)
            context_1_first = context_1.gather(
                dim=1, index=start_1.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.hidden_dim)
            )
            context_1_last = context_1.gather(
                dim=1, index=end_1.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.hidden_dim)
            )
            context_2_first = context_2.gather(
                dim=1, index=start_2.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.hidden_dim)
            )
            context_2_last = context_2.gather(
                dim=1, index=end_2.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.hidden_dim)
            )

            # (batch, seq_len*, num_perspectives)
            matching_vector_1_forward_full = multi_perspective_match(
                context_1, context_2_last, self.full_forward_match_weights
            )
            matching_vector_2_forward_full = multi_perspective_match(
                context_2, context_1_last, self.full_forward_match_weights_reversed
            )
            matching_vector_1_backward_full = multi_perspective_match(
                context_1, context_2_first, self.full_backward_match_weights
            )
            matching_vector_2_backward_full = multi_perspective_match(
                context_2, context_1_first, self.full_backward_match_weights_reversed
            )

            matching_vector_1.extend(matching_vector_1_forward_full)
            matching_vector_1.extend(matching_vector_1_backward_full)
            matching_vector_2.extend(matching_vector_2_forward_full)
            matching_vector_2.extend(matching_vector_2_backward_full)

        # Step 2. Maxpooling-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with every time step of the forward (or backward)
        # contextual embedding of the other sentence, and only the max value of each
        # dimension is retained.
        if self.with_maxpool_match:
            # (batch, seq_len1, seq_len2, num_perspectives)
            matching_vector_max = multi_perspective_match_pairwise(
                context_1, context_2, self.maxpool_match_weights
            )

            # (batch, seq_len*, num_perspectives)
            matching_vector_1_max = masked_max(
                matching_vector_max, mask_2.unsqueeze(-2).unsqueeze(-1), dim=2
            )
            matching_vector_1_mean = masked_mean(
                matching_vector_max, mask_2.unsqueeze(-2).unsqueeze(-1), dim=2
            )
            matching_vector_2_max = masked_max(
                matching_vector_max.permute(0, 2, 1, 3), mask_1.unsqueeze(-2).unsqueeze(-1), dim=2
            )
            matching_vector_2_mean = masked_mean(
                matching_vector_max.permute(0, 2, 1, 3), mask_1.unsqueeze(-2).unsqueeze(-1), dim=2
            )

            matching_vector_1.extend([matching_vector_1_max, matching_vector_1_mean])
            matching_vector_2.extend([matching_vector_2_max, matching_vector_2_mean])

        # Step 3. Attentive-Matching
        # Each forward (or backward) similarity is taken as the weight
        # of the forward (or backward) contextual embedding, and calculate an
        # attentive vector for the sentence by weighted summing all its
        # contextual embeddings.
        # Finally match each forward (or backward) contextual embedding
        # with its corresponding attentive vector.

        # (batch, seq_len1, seq_len2, hidden_dim)
        att_2 = context_2.unsqueeze(-3) * cosine_sim.unsqueeze(-1)

        # (batch, seq_len1, seq_len2, hidden_dim)
        att_1 = context_1.unsqueeze(-2) * cosine_sim.unsqueeze(-1)

        if self.with_attentive_match:
            # (batch, seq_len*, hidden_dim)
            att_mean_2 = masked_softmax(
                att_2.sum(dim=2), mask_1.unsqueeze(-1)
            )
            att_mean_1 = masked_softmax(
                att_1.sum(dim=1), mask_2.unsqueeze(-1)
            )

            # (batch, seq_len*, num_perspectives)
            matching_vector_1_att_mean = multi_perspective_match(
                context_1, att_mean_2, self.attentive_match_weights
            )
            matching_vector_2_att_mean = multi_perspective_match(
                context_2, att_mean_1, self.attentive_match_weights_reversed
            )
            matching_vector_1.extend(matching_vector_1_att_mean)
            matching_vector_2.extend(matching_vector_2_att_mean)

        # Step 4. Max-Attentive-Matching
        # Pick the contextual embeddings with the highest cosine similarity as the attentive
        # vector, and match each forward (or backward) contextual embedding with its
        # corresponding attentive vector.
        if self.with_max_attentive_match:
            # (batch, seq_len*, hidden_dim)
            att_max_2 = masked_max(
                att_2, mask_2.unsqueeze(-2).unsqueeze(-1), dim=2
            )
            att_max_1 = masked_max(
                att_1.permute(0, 2, 1, 3), mask_1.unsqueeze(-2).unsqueeze(-1), dim=2
            )

            # (batch, seq_len*, num_perspectives)
            matching_vector_1_att_max = multi_perspective_match(
                context_1, att_max_2, self.max_attentive_match_weights
            )
            matching_vector_2_att_max = multi_perspective_match(
                context_2, att_max_1, self.max_attentive_match_weights_reversed
            )

            matching_vector_1.extend(matching_vector_1_att_max)
            matching_vector_2.extend(matching_vector_2_att_max)

        return matching_vector_1, matching_vector_2

    @staticmethod
    def multi_perspective_match(vector1, vector2, weight):
        assert vector1.size(0) == vector2.size(0)
        assert weight.size(1) == vector1.size(2)

        # (batch, seq_len, 1)
        similarity_single = F.cosine_similarity(vector1, vector2, 2).unsqueeze(2)

        # (1, 1, num_perspectives, hidden_size)
        weight = weight.unsqueeze(0).unsqueeze(0)

        # (batch, seq_len, num_perspectives, hidden_size)
        vector1 = weight * vector1.unsqueeze(2)
        vector2 = weight * vector2.unsqueeze(2)

        similarity_multi = F.cosine_similarity(vector1, vector2, dim=3)

        return similarity_single, similarity_multi

    @staticmethod
    def multi_perspective_match_pairwise(vector1, vector2, weight):
        num_perspectives = weight.size(0)

        # (1, num_perspectives, 1, hidden_size)
        weight = weight.unsqueeze(0).unsqueeze(2)

        # (batch, num_perspectives, seq_len*, hidden_size)
        vector1 = weight * vector1.unsqueeze(1).expand(-1, num_perspectives, -1, -1)
        vector2 = weight * vector2.unsqueeze(1).expand(-1, num_perspectives, -1, -1)

        # (batch, num_perspectives, seq_len*, 1)
        vector1_norm = vector1.norm(p=2, dim=3, keepdim=True)
        vector2_norm = vector2.norm(p=2, dim=3, keepdim=True)

        # (batch, num_perspectives, seq_len1, seq_len2)
        mul_result = th.matmul(vector1, vector2.transpose(2, 3))
        norm_value = vector1_norm * vector2_norm.transpose(2, 3)

        # (batch, seq_len1, seq_len2, num_perspectives)
        return (mul_result / norm_value.clamp(min=EPS)).permute(0, 2, 3, 1)

    def get_output_dim(self):
        return self.output_dim


class Unsqueeze(nn.Module):
    def __init__(self, dim=-1):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class Highway(nn.Module):
    def __init__(self, input_dim, num_layers=1, activation="relu"):
        super(Highway, self).__init__()

        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for i in range(num_layers)])
        self.act = map_activation_str_to_layer(activation)

        # init
        for layer in self.layers:
            init_weight(layer, activation=activation)
            nn.init.zeros_(layer.bias[:input_dim])
            nn.init.ones_(layer.bias[input_dim:])

    def forward(self, x):
        for layer in self.layers:
            o, g = layer(x).chunk(2, dim=-1)
            o = self.act(o)
            g = F.sigmoid(g)
            x = g * x + (1 - g) * o
        return x


class CnnHighway(nn.Module):
    def __init__(self, input_dim, filters, num_highway=1, activation="relu", norm_layer="none", dropout=0.0):
        super(CnnHighway, self).__init__()

        self.input_dim = input_dim

        # Create the convolutions
        self.convs = []
        for i, (width, num) in enumerate(filters):
            # speed up by putting activation after maxpool
            conv = Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=num, kernel_size=width, bias=True),
                nn.AdaptiveMaxPool1d(1),
                map_activation_str_to_layer(activation),
                Unsqueeze(2)
            )
            self.convs.append(conv)
        self.convs = Parallel(*self.convs)

        # Create the highway layers
        self.output_dim = sum(num for _, num in filters)
        self.highways = Highway(self.output_dim, num_highway, activation=activation)

        # And add a layer norm
        if norm_layer == "none":
            self.norm_layer = None
        elif norm_layer == "batch_norm":
            self.norm_layer = nn.BatchNorm1d(self.output_dim)
        elif norm_layer == "layer_norm":
            self.norm_layer = nn.LayerNorm(self.output_dim)
        elif norm_layer == "group_norm":
            self.norm_layer = nn.GroupNorm(32, self.output_dim)
        else:
            raise ValueError

        self.drop = nn.Dropout(dropout)

        # init
        for layer in self.convs:
            init_weight(layer, activation=activation)
        if self.norm_layer is not None:
            init_weight(self.norm_layer)

    def forward(self, x):
        x = x.transpose(1, 2)
        output = self.convs(x)
        output = self.highways(output)

        if self.norm_layer is not None:
            output = self.norm_layer(output)

        output = self.drop(output)

        return output

    def get_output_dim(self):
        return self.output_dim


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_mlp_layers=2,
        activation="none",
        norm_layer="none"
    ):
        super(MLP, self).__init__()

        self.num_mlp_layers = num_mlp_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = []
        for i in range(num_mlp_layers):
            if i == 0:
                if num_mlp_layers > 1:
                    self.layers.append(nn.Linear(input_dim, hidden_dim))
                else:
                    self.layers.append(nn.Linear(input_dim, output_dim))
            else:
                if norm_layer == "none":
                    pass
                elif norm_layer == "batch_norm":
                    self.layers.append(nn.BatchNorm1d(hidden_dim))
                elif norm_layer == "layer_norm":
                    self.layers.append(nn.LayerNorm(hidden_dim))
                elif norm_layer == "group_norm":
                    self.layers.append(nn.GroupNorm(32, hidden_dim))
                else:
                    raise ValueError
                self.layers.append(map_activation_str_to_layer(activation))
                if i < num_mlp_layers - 1:
                    self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                else:
                    self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*self.layers)

        # init
        for layer in self.layers:
            init_weight(layer, activation=activation, init="uniform")

    def forward(self, x):
        return self.layers(x)

    def get_output_dim(self):
        return self.output_dim