import torch as th
import torch.nn as nn
from layer import *


class MLPPredictor(nn.Module):
    def __init__(self, **kw):
        super(MLPPredictor, self).__init__()
        input_dim = kw.get("input_dim", 14)
        hidden_dim = kw.get("hidden_dim", 128)
        num_layers = kw.get("num_layers", 1)
        num_segments = kw.get("num_segments", 2)
        num_labels = kw.get("num_labels", 3)
        dropout = kw.get("dropout", 0.0)
        activation = kw.get("activation", "tanh")

        self.fc_layer = MLP(
            input_dim=input_dim*num_segments*2,
            hidden_dim=hidden_dim,
            output_dim=num_labels,
            num_mlp_layers=2,
            activation=activation,
            norm_layer="batch_norm"
        )

    def forward(self, x, mask):
        bsz, x_len, x_dim = x.size()
        x = x.masked_fill((mask == 0).unsqueeze(-1), 0.0)
        x = th.cat([x, x.softmax(dim=-1)], dim=-1)
        pad_len = self.fc_layer.input_dim//x_dim//2 - x_len
        h = th.cat(
            [th.zeros((bsz, pad_len, x_dim*2), dtype=x.dtype, device=x.device), x],
            dim=1
        )
        h = h.view(bsz, -1)
        output = self.fc_layer(h)

        return output


class CNNPredictor(nn.Module):
    def __init__(self, **kw):
        super(CNNPredictor, self).__init__()
        input_dim = kw.get("input_dim", 14)
        hidden_dim = kw.get("hidden_dim", 128)
        num_layers = kw.get("num_layers", 1)
        conv_filters = kw.get("conv_filters", 64)
        num_segments = kw.get("num_segments", 2)
        num_labels = kw.get("num_labels", 3)
        dropout = kw.get("dropout", 0.0)
        activation = kw.get("activation", "tanh")

        self.special_vec = th.empty((1, 1, 2*input_dim), dtype=th.float)
        self.pos_emb = PositionEmbedding(2*input_dim, num_segments+2, scale=1/input_dim**0.5)
        self.cnn = CnnHighway(
            input_dim=2*input_dim,
            filters=[(1, conv_filters), (2, conv_filters)],
            num_highway=num_layers,
            activation=activation,
            layer_norm=True,
            dropout=dropout
        )
        dim = self.cnn.get_output_dim()
        self.fc_layer = MLP(
            input_dim=dim,
            hidden_dim=hidden_dim,
            output_dim=num_labels,
            num_mlp_layers=2,
            activation=activation,
            norm_layer="batch_norm"
        )

        # init
        nn.init.normal_(self.special_vec[:, :, :input_dim], 0.0, 1/input_dim**0.5)
        nn.init.normal_(self.special_vec[:, :, input_dim:], 1/input_dim)

    def forward(self, x, mask):
        bsz = x.size(0)
        x = x.masked_fill((mask == 0).unsqueeze(-1), 0.0)
        x = th.cat([x, x.softmax(dim=-1)], dim=-1)
        h = th.cat([x, self.special_vec.expand(bsz, -1, -1)], dim=1)
        h_mask = th.cat([mask, th.ones((bsz, 1), dtype=th.bool, device=mask.device)], dim=1)
        cumsum = th.cumsum(h_mask, dim=1)
        pos_emb = self.pos_emb(cumsum)
        h = h + pos_emb

        h = self.cnn(h)
        output = self.fc_layer(h)

        return output


class LSTMPredictor(nn.Module):
    def __init__(self, **kw):
        super(LSTMPredictor, self).__init__()
        input_dim = kw.get("input_dim", 14)
        hidden_dim = kw.get("hidden_dim", 128)
        num_layers = kw.get("num_layers", 1)
        num_segments = kw.get("num_segments", 2)
        num_labels = kw.get("num_labels", 3)
        dropout = kw.get("dropout", 0.0)
        activation = kw.get("activation", "tanh")

        self.special_vec = th.empty((1, 1, 2*input_dim), dtype=th.float)
        self.pos_emb = PositionEmbedding(2*input_dim, num_segments+2, scale=1/input_dim**0.5)
        self.lstm = nn.LSTM(
            2*input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.fc_layer = MLP(
            input_dim=2*hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_labels,
            num_mlp_layers=2,
            activation=activation,
            norm_layer="batch_norm"
        )

        # init
        nn.init.normal_(self.special_vec[:, :, :input_dim], 0.0, 1/input_dim**0.5)
        nn.init.normal_(self.special_vec[:, :, input_dim:], 1/input_dim)

    def forward(self, x, mask):
        bsz = x.size(0)
        x = x.masked_fill((mask == 0).unsqueeze(-1), 0.0)
        x = th.cat([x, x.softmax(dim=-1)], dim=-1)
        h = th.cat([x, self.special_vec.expand(bsz, -1, -1)], dim=1)
        h_mask = th.cat([mask, th.ones((bsz, 1), dtype=th.bool, device=mask.device)], dim=1)
        cumsum = th.cumsum(h_mask, dim=1)
        pos_emb = self.pos_emb(cumsum)
        h = h + pos_emb

        _, (h, _) = self.lstm(h)
        h = h.transpose(0, 1).contiguous().view(bsz, self.lstm.num_layers, -1)
        h = h[:, -1]
        output = self.fc_layer(h)

        return output


class TransformerPredictor(nn.Module):
    def __init__(self, **kw):
        super(TransformerPredictor, self).__init__()
        input_dim = kw.get("input_dim", 14)
        hidden_dim = kw.get("hidden_dim", 128)
        num_layers = kw.get("num_layers", 1)
        num_perspectives = kw.get("num_perspectives", 8)
        num_segments = kw.get("num_segments", 2)
        num_labels = kw.get("num_labels", 3)
        dropout = kw.get("dropout", 0.0)
        activation = kw.get("activation", "tanh")

        self.special_vec = th.empty((1, 1, 2*input_dim), dtype=th.float)
        self.pos_emb = PositionEmbedding(2*input_dim, num_segments+2, scale=1/input_dim**0.5)
        self.attns = []
        for i in range(num_layers):
            self.attns.append(
                TransformerLayer(
                    input_dim=2*input_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_perspectives,
                    dropout=dropout
                )
            )
        self.attns = nn.ModuleList(self.attns)
        self.fc_layer = MLP(
            input_dim=2*input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_labels,
            num_mlp_layers=2,
            activation=activation,
            norm_layer="batch_norm"
        )

        # init
        nn.init.normal_(self.special_vec[:, :, :input_dim], 0.0, 1/input_dim**0.5)
        nn.init.normal_(self.special_vec[:, :, input_dim:], 1/input_dim)

    def forward(self, x, mask):
        bsz = x.size(0)
        x = x.masked_fill((mask == 0).unsqueeze(-1), 0.0)
        x = th.cat([x, x.softmax(dim=-1)], dim=-1)
        h = th.cat([x, self.special_vec.expand(bsz, -1, -1)], dim=1)
        h_mask = th.cat([mask, th.ones((bsz, 1), dtype=th.bool, device=mask.device)], dim=1)
        cumsum = th.cumsum(h_mask, dim=1)
        pos_emb = self.pos_emb(cumsum)
        h = h + pos_emb

        for i in range(len(self.attns) - 1):
            h = self.attns[i](h, h, h, h_mask, h_mask)
        h = self.attns[-1](h[:, -1:], h, h, key_mask=h_mask)
        h = h.view(bsz, -1)
        output = self.fc_layer(h)

        return output
