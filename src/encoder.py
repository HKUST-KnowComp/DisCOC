import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os
from dataset import DISCO2ID, STANCE2ID, INF
from tokenizer import *
from util import split_ids, init_weight
from util import WORD_EMB_MEAN, WORD_EMB_STD, TYPE_EMB_MEAN, TYPE_EMB_STD


INIT_EMB_MEAN = 0.0
INIT_EMB_STD = 0.01


def load_wordvec(path, word2idx=None):
    if word2idx is None:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            line = line.split(" ", 1)
            try:
                # num_word, dim
                num_word = int(line[0])
                embed_dim = int(line[1])
            except ValueError:
                word = line[0]
                word2idx[word] = len(word2idx)
            for line in f:
                word = line.split(" ", 1)[0]
                word2idx[word] = len(word2idx)
        return load_wordvec(path, word2idx=word2idx)

    pad_idx = word2idx.get(PAD, -1)
    unk_idx = word2idx.get(UNK, -1)
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
        line = line.strip().split(" ", 1)
        try:
            # num_word, dim
            embed_dim = int(line[1])
            weight = np.zeros((len(word2idx), embed_dim), dtype=np.float32)
            count = np.zeros((len(word2idx),), dtype=np.int64)
        except ValueError:
            word, vec = line
            embed_dim = len(vec.split())
            weight = np.zeros((len(word2idx), embed_dim), dtype=np.float32)
            count = np.zeros((len(word2idx),), dtype=np.int64)
            word_idx = word2idx.get(word, unk_idx)
            if word_idx >= 0:
                weight[word_idx] += np.array(vec.split(), dtype=np.float32)
                count[word_idx] += 1
        for line in f:
            word, vec = line.strip().split(" ", 1)
            word_idx = word2idx.get(word, unk_idx)
            if word_idx >= 0:
                weight[word_idx] += np.array(vec.split(), dtype=np.float32)
                count[word_idx] += 1
    count = np.clip(count, a_min=1, a_max=None)
    weight = weight / count.reshape(-1, 1)
    weight[pad_idx] = 0.0

    return weight, word2idx


def refine_emb(sent_ids, original_emb, dummy_emb=None):
    bsz, x_len = sent_ids.size()

    if dummy_emb is None:
        dummy_emb = th.zeros(original_emb.size(-1), dtype=original_emb.dtype, device=original_emb.device)
    refined_emb = []
    for i in range(bsz):
        sent_indices = split_ids(sent_ids[i])
        if sent_indices.size(0) == 1:
            refined_emb.append(dummy_emb.unsqueeze(0).repeat(x_len, 1))
        else:
            real_emb = th.cat(
                [
                    th.index_select(original_emb[i], dim=0, index=sent_indices[:-1]),
                    dummy_emb.unsqueeze(0)
                ],
                dim=0
            )
            refined_emb.append(th.index_select(real_emb, dim=0, index=sent_ids[i]-sent_ids[i][0]))
    refined_emb = th.stack(refined_emb, dim=0)

    return refined_emb


class Encoder(nn.Module):
    def __init__(self, **kw):
        super(Encoder, self).__init__()

        self.special_token_ids = dict()
        init_tau = kw.get("init_tau", 1.0)
        num_stances = len(STANCE2ID)
        num_discos = len(DISCO2ID)

        self.num_stances = num_stances
        self.num_discos = num_discos
        embed_dim = self.get_embeding_dim()
        self.stance_t = nn.Parameter(th.Tensor([init_tau]))
        self.stance_w1 = nn.Parameter(th.Tensor(num_stances, embed_dim))
        self.stance_w2 = nn.Parameter(th.Tensor(num_stances, embed_dim))
        self.disco_t = nn.Parameter(th.Tensor([init_tau]))
        self.disco_w1 = nn.Parameter(th.Tensor(num_discos, embed_dim))
        self.disco_w2 = nn.Parameter(th.Tensor(num_discos, embed_dim))

        # init
        with th.no_grad():
            nn.init.constant_(self.stance_t, init_tau)
            nn.init.constant_(self.disco_t, init_tau)
            nn.init.normal_(self.stance_w1, INIT_EMB_MEAN, INIT_EMB_STD)
            nn.init.constant_(self.stance_w1[0], 0.0)
            nn.init.normal_(self.stance_w2, INIT_EMB_MEAN, INIT_EMB_STD)
            nn.init.constant_(self.stance_w2[0], 0.0)
            nn.init.normal_(self.disco_w1, INIT_EMB_MEAN, INIT_EMB_STD)
            nn.init.constant_(self.disco_w1[0], 0.0)
            nn.init.normal_(self.disco_w2, INIT_EMB_MEAN, INIT_EMB_STD)
            nn.init.constant_(self.disco_w2[0], 0.0)

    def set_finetune(self, finetune):
        raise NotImplementedError

    def load_pt(self, model_path=None):
        device = next(iter(self.parameters())).device
        state_dict = th.load(model_path, map_location=device)
        model_state_dict_keys = self.state_dict().keys()
        assert len(state_dict) == len(model_state_dict_keys)

        if len(set(model_state_dict_keys) & set(state_dict.keys())) == len(model_state_dict_keys):
            self.load_state_dict(state_dict)
        else:
            self.load_state_dict(dict(zip(model_state_dict_keys, state_dict.values())))

    def forward(self, x, mask=None, type_ids=None, pos_ids=None, mems=None, mems_mask=None):
        raise NotImplementedError

    def init_embeddings(self, num_words, num_segments=1, **kw):
        init_tau = kw.get("init_tau", 1.0)
        with th.no_grad():
            if hasattr(self, "stance_t"):
                nn.init.constant_(self.stance_t, init_tau)
            if hasattr(self, "disco_t"):
                nn.init.constant_(self.disco_t, init_tau)
            if hasattr(self, "stance_w1"):
                nn.init.normal_(self.stance_w1, INIT_EMB_MEAN, INIT_EMB_STD)
                nn.init.constant_(self.stance_w1[0], 0.0)
            if hasattr(self, "stance_w2"):
                nn.init.normal_(self.stance_w2, INIT_EMB_MEAN, INIT_EMB_STD)
                nn.init.constant_(self.stance_w2[0], 0.0)
            if hasattr(self, "disco_w1"):
                nn.init.normal_(self.disco_w1, INIT_EMB_MEAN, INIT_EMB_STD)
                nn.init.constant_(self.disco_w1[0], 0.0)
            if hasattr(self, "disco_w2"):
                nn.init.normal_(self.disco_w2, INIT_EMB_MEAN, INIT_EMB_STD)
                nn.init.constant_(self.disco_w2[0], 0.0)

    def get_special_token_id(self, key):
        return self.special_token_ids[key]

    def get_special_rep_idx(self, x, special_token=CLS):
        special_token_id = self.get_special_token_id(special_token)
        if th.all(x[..., 0] == special_token_id):
            return 0
        if th.all(x[..., (x.size(-1) - 1)] == special_token_id):
            return x.size(-1) - 1
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError

    def get_embeding_dim(self):
        raise NotImplementedError

    def extra_repr(self):
        summary = []
        for x in ["stance_t", "stance_w1", "stance_w2", "disco_t", "disco_w1", "disco_w2"]:
            param = getattr(self, x)
            summary.append(
                "(%s): Parameter(size=%s, requires_grad=%s)" % (
                    x, tuple(param.size()), param.requires_grad
                )
            )
        return "\n".join(summary)


class LSTMEncoder(Encoder):
    def __init__(self, **kw):
        super(LSTMEncoder, self).__init__(**kw)

        word2vec_file = kw.get("word2vec_file", "../data/model/glove/glove_kialo.txt")
        word_dim = kw.get("word_dim", 300)
        hidden_dim = kw.get("hidden_dim", 128)
        num_layers = kw.get("num_layers", 1)
        num_segments = kw.get("num_segments", 1)
        dropout = kw.get("dropout", 0.2)
        special_tokens = kw.get("special_tokens", list())

        # embedding layers
        tokenizer = SpacyTokenizer(special_tokens=special_tokens, word2vec_file=word2vec_file)
        self.special_token_ids = {
            token: tokenizer.get_special_token_id(token)
            for token in tokenizer.special_tokens
        }

        self.word_embeddings = nn.Embedding(
            len(tokenizer.word2idx), word_dim,
            padding_idx=tokenizer.get_special_token_id(PAD)
        )
        self.token_type_embeddings = nn.Embedding(1, word_dim)

        # LSTM layers
        self.model = nn.LSTM(
            word_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # init
        with th.no_grad():
            self.init_embeddings(
                num_words=len(tokenizer.word2idx),
                num_segments=num_segments,
                word2idx=tokenizer.word2idx,
                word2vec_file=word2vec_file,
            )
            nn.init.normal_(self.token_type_embeddings.weight, TYPE_EMB_MEAN, TYPE_EMB_STD)
            init_weight(self.model, activation="tanh")

        self.set_finetune("full")

    def init_embeddings(self, num_words, num_segments=1, **kw):
        super(LSTMEncoder, self).init_embeddings(num_words, num_segments, **kw)

        word2idx = kw.get("word2idx", None)
        if word2idx is None:
            print("Warning: word2idx is None. We would regard the line no as the word index.")
        word2vec_file = kw.get("word2vec_file", None)

        num_word_embeddings = self.word_embeddings.num_embeddings
        num_token_type_embeddings = self.token_type_embeddings.num_embeddings
        embedding_dim = self.word_embeddings.embedding_dim

        with th.no_grad():
            # word embeddings
            if num_words > num_word_embeddings:
                new_word_embeddings = nn.Embedding(
                    num_words, embedding_dim,
                    padding_idx=self.word_embeddings.padding_idx
                )
                new_word_embeddings.weight.data[:num_word_embeddings].copy_(
                    self.word_embeddings.weight
                )
                new_word_embeddings.weight.data[num_word_embeddings:].copy_(
                    self.word_embeddings.weight[self.get_special_token_id(UNK)].unsqueeze(0)
                )
                del self.word_embeddings
                self.word_embeddings = new_word_embeddings

            if word2vec_file is not None:
                wordvec, word2idx = load_wordvec(word2vec_file, word2idx)
                with th.no_grad():
                    self.word_embeddings.data.copy_(th.from_numpy(wordvec))

            # segment embeddings
            r = math.ceil(num_segments / num_token_type_embeddings)
            new_token_type_embeddings = nn.Embedding(r * num_token_type_embeddings, embedding_dim)
            new_token_type_embeddings.weight.data.copy_(
                self.token_type_embeddings.weight.repeat(r, 1)
            )

            del self.token_type_embeddings
            self.token_type_embeddings = new_token_type_embeddings

    def set_finetune(self, finetune):
        if finetune == "none":
            for param in self.parameters():
                param.requires_grad = False
        elif finetune == "full":
            for param in self.parameters():
                param.requires_grad = True
        elif finetune == "layers":
            for param in self.parameters():
                param.requires_grad = False
            for param in self.model.parameters():
                param.requires_grad = True
        elif finetune == "last":
            for param in self.parameters():
                param.requires_grad = False
            for param in self.model.lstms[-1].parameters():
                param.requires_grad = True
        elif finetune == "type":
            for param in self.parameters():
                param.requires_grad = False
            self.token_type_embeddings.weight.requires_grad = True
        else:
            raise NotImplementedError("Error: finetune=%s is not supported!" % (finetune))

        self.word_embeddings.weight.requires_grad = False

    def forward(self, x, mask=None, type_ids=None, pos_ids=None, mems=None, mems_mask=None):
        if type_ids is None:
            type_ids = th.zeros_like(x)
        while type_ids.dim() < x.dim():
            type_ids = type_ids.unsqueeze(0)

        word_emb = self.word_embeddings(x)
        token_type_emb = self.token_type_embeddings(type_ids)

        emb = word_emb + token_type_emb

        output, new_mems = self.model(emb, mems)

        return output, new_mems

    def get_output_dim(self):
        return self.model.hidden_size * 2

    def get_embeding_dim(self):
        raise self.word_embeddings.embedding_dim


class TransformerEncoder(Encoder):
    def __init__(self, **kw):
        super(TransformerEncoder, self).__init__(**kw)

        self.max_len = kw.get("max_len", 1024)
        self.mem_len = kw.get("mem_len", self.max_len)


    def init_embeddings(self, num_words, num_segments, **kw):
        super(TransformerEncoder, self).init_embeddings(num_words, num_segments, **kw)

        embedding_dim = self.model.embeddings.word_embeddings.embedding_dim
        num_word_embeddings = self.model.embeddings.word_embeddings.num_embeddings
        num_position_embeddings = self.model.embeddings.position_embeddings.num_embeddings
        num_token_type_embeddings = self.model.embeddings.token_type_embeddings.num_embeddings

        with th.no_grad():
            # word embeddings
            if num_words > num_word_embeddings:
                new_word_embeddings = nn.Embedding(
                    num_words, embedding_dim,
                    padding_idx=self.model.embeddings.word_embeddings.padding_idx
                )
                new_word_embeddings.weight.data[:num_word_embeddings].copy_(
                    self.model.embeddings.word_embeddings.weight
                )
                new_word_embeddings.weight.data[num_word_embeddings:].copy_(
                    self.model.embeddings.word_embeddings.weight[self.get_special_token_id(UNK)].unsqueeze(0)
                )
                del self.model.embeddings.word_embeddings
                self.model.embeddings.word_embeddings = new_word_embeddings

            # position embeddings
            if self.max_len * num_segments > num_position_embeddings:
                r = math.ceil(
                    self.max_len * num_segments / num_position_embeddings
                )
                new_position_embeddings = nn.Embedding(
                    r * num_position_embeddings, embedding_dim,
                    padding_idx=self.model.embeddings.position_embeddings.padding_idx
                )
                original_pos_emb = self.model.embeddings.position_embeddings.weight
                flipped_pos_emb = th.flip(original_pos_emb, (0, ))
                for i in range(r):
                    pos_idx = i * num_position_embeddings
                    new_position_embeddings.weight.data[pos_idx: (pos_idx + num_position_embeddings)].copy_(
                        original_pos_emb if i % 2 == 0 else flipped_pos_emb
                    )
                if hasattr(self.model.embeddings, "position_ids"):
                    self.model.embeddings.position_ids = th.arange(r * num_position_embeddings).expand((1, -1))
                del self.model.embeddings.position_embeddings
                self.model.embeddings.position_embeddings = new_position_embeddings

            # segment embeddings
            if num_segments > num_token_type_embeddings:
                r = math.ceil(
                    num_segments / num_token_type_embeddings
                )
                new_token_type_embeddings = nn.Embedding(
                    r * num_token_type_embeddings, embedding_dim,
                    padding_idx=self.model.embeddings.token_type_embeddings.padding_idx
                )
                new_token_type_embeddings.weight.data.copy_(
                    self.model.embeddings.token_type_embeddings.weight.repeat(r, 1)
                )
                del self.model.embeddings.token_type_embeddings
                self.model.embeddings.token_type_embeddings = new_token_type_embeddings

    def set_finetune(self, finetune):
        if finetune == "none":
            for param in self.parameters():
                param.requires_grad = False
        elif finetune == "full":
            for param in self.parameters():
                param.requires_grad = True
        elif finetune == "layers":
            for param in self.parameters():
                param.requires_grad = False
            for param in self.model.encoder.parameters():
                param.requires_grad = True
        elif finetune == "last":
            for param in self.parameters():
                param.requires_grad = False
            for param in self.model.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif finetune == "type":
            for param in self.parameters():
                param.requires_grad = False
            self.model.embeddings.token_type_embeddings.weight.requires_grad = True
        else:
            raise NotImplementedError("Error: finetune=%s is not supported!" % (finetune))

    def forward(
            self,
            x,
            mask=None,
            sent_ids=None,
            type_ids=None,
            pos_ids=None,
            stance_logit=None,
            disco_logit=None,
            mems=None,
            mems_mask=None
    ):
        bsz, x_len = x.size()
        device = x.device

        if mask is None:
            mask = th.ones_like(x)
        if sent_ids is None:
            sent_ids = th.zeros_like(x)
        elif sent_ids.size() != x.size():
            sent_ids = sent_ids.unsqueeze(0)
        if type_ids is None:
            type_ids = th.zeros_like(x)
        elif type_ids.size() != x.size():
            type_ids = type_ids.unsqueeze(0)

        if mems is None:
            mlen = 0
            mems = [None] * self.model.config.num_hidden_layers
            extended_mask = self.model.get_extended_attention_mask(mask, (bsz, x_len), device)
        else:
            mlen = mems[0].size(1)
            if mems_mask is None:
                mems_mask = th.ones_like(mems[0]).to(mask)
            extended_mask = th.cat([mems_mask, mask], dim=1)
            extended_mask = self.model.get_extended_attention_mask(extended_mask, (bsz, mlen + x_len), device)

        word_emb = self.model.embeddings(input_ids=x, position_ids=pos_ids, token_type_ids=type_ids)

        dummy_stance_logit = th.zeros((self.stance_w2.size(0),), device=self.stance_w2.device,
                                      dtype=self.stance_w2.dtype)
        dummy_stance_logit[0].fill_(INF)
        if stance_logit is None:
            stance_logit = dummy_stance_logit.view(1, 1, -1).expand(bsz, x_len, -1)
        stance_weight = F.softmax(stance_logit / self.stance_t, dim=-1)
        stance_emb1 = th.matmul(stance_weight, self.stance_w1)
        dummy_stance_emb1 = self.stance_w1[0]
        refined_stance_emb1 = refine_emb(sent_ids, stance_emb1, dummy_stance_emb1)
        stance_emb2 = th.matmul(stance_weight, self.stance_w2)

        dummy_disco_logit = th.zeros((self.disco_w2.size(0),), device=self.disco_w2.device, dtype=self.disco_w2.dtype)
        dummy_disco_logit[0].fill_(INF)
        if disco_logit is None:
            disco_logit = dummy_disco_logit.view(1, 1, -1).expand(bsz, x_len, -1)
        disco_weight = F.softmax(disco_logit / self.disco_t, dim=-1)
        disco_emb1 = th.matmul(disco_weight, self.disco_w1)
        dummy_disco_emb1 = self.disco_w1[0]
        refined_disco_emb1 = refine_emb(sent_ids, disco_emb1, dummy_disco_emb1)
        disco_emb2 = th.matmul(disco_weight, self.disco_w2)

        disco_stance_emb = refined_stance_emb1 + stance_emb2 + refined_disco_emb1 + disco_emb2
        hidden = word_emb + disco_stance_emb

        if mask.dim() == 2:
            zero_mask = (mask == 0).unsqueeze(-1)
        else:
            zero_mask = (th.diagonal(mask, dim1=1, dim2=2) == 0).unsqueeze(-1)
        hidden.masked_fill_(zero_mask, 0.0)

        new_mems = []
        for i, layer in enumerate(self.model.encoder.layer):
            if self.mem_len > 0:
                new_mems.append(self.cache_mem(hidden, mems[i]))
            else:
                new_mems.append(None)

            if mems[i] is None:
                extended_hidden = hidden
                hidden = layer(hidden, extended_mask)[0]
            else:
                extended_hidden = th.cat([mems[i], hidden], dim=1)
                hidden = layer(extended_hidden, extended_mask)[0][:, -x_len:]
            hidden.masked_fill_(zero_mask, 0.0)

        if hidden.is_contiguous():
            output = hidden
        else:
            output = hidden.contiguous()
        new_mems = tuple(new_mems)

        return output, new_mems

    def cache_mem(self, curr_out, prev_mem):
        if self.mem_len == 0:
            # If :obj:`use_mems` is active but no `mem_len` is defined, the model behaves like GPT-2 at inference time
            # and returns all of the past and current hidden states.
            cutoff = 0
        else:
            # If :obj:`use_mems` is active and `mem_len` is defined, the model returns the last `mem_len` hidden
            # states. This is the preferred setting for training and long-form generation.
            cutoff = -self.mem_len
        if prev_mem is None:
            # if :obj:`use_mems` is active and `mem_len` is defined, the model
            new_mem = curr_out[:, cutoff:]  # bsz x seq_len x dim
        else:
            new_mem = th.cat([prev_mem, curr_out], dim=1)[:, cutoff:]

        return new_mem.detach()

    def get_output_dim(self):
        return self.model.config.hidden_size

    def get_embeding_dim(self):
        return 768


class BertEncoder(TransformerEncoder):
    def __init__(self, **kw):
        super(BertEncoder, self).__init__(**kw)

        from transformers import BertModel
        from tokenizer import BertTokenizer

        bert_dir = kw.get("bert_dir", os.path.join(DOWNLOAD_DIR, "pretrained_lm", "bert"))
        special_tokens = kw.get("special_tokens", list())

        tokenizer = BertTokenizer(special_tokens=special_tokens, bert_dir=bert_dir)
        self.special_token_ids = {
            token: tokenizer.get_special_token_id(token)
            for token in tokenizer.special_tokens
        }

        self.model = BertModel.from_pretrained("bert-base-uncased", cache_dir=bert_dir)

        self.init_embeddings(
            num_words=max(
                max(self.special_token_ids.values()) + 1,
                self.model.embeddings.word_embeddings.num_embeddings
            ),
            num_segments=kw.get(
                "num_segments",
                2
            )
        )

        self.set_finetune("full")


class RobertaEncoder(TransformerEncoder):
    def __init__(self, **kw):
        super(RobertaEncoder, self).__init__(**kw)

        from transformers import RobertaModel
        from tokenizer import RobertaTokenizer

        roberta_dir = kw.get("roberta_dir", os.path.join(DOWNLOAD_DIR, "pretrained_lm", "roberta"))
        special_tokens = kw.get("special_tokens", list())

        tokenizer = RobertaTokenizer(special_tokens=special_tokens, roberta_dir=roberta_dir)
        self.special_token_ids = {
            token: tokenizer.get_special_token_id(token)
            for token in tokenizer.special_tokens
        }

        self.model = RobertaModel.from_pretrained("roberta-base", cache_dir=roberta_dir)

        self.init_embeddings(
            num_words=max(
                max(self.special_token_ids.values()) + 1,
                self.model.embeddings.word_embeddings.num_embeddings
            ),
            num_segments=kw.get(
                "num_segments",
                1
            )
        )

        self.set_finetune("full")


class AlbertEncoder(TransformerEncoder):
    def __init__(self, **kw):
        super(AlbertEncoder, self).__init__(**kw)

        from transformers import AlbertModel
        from tokenizer import AlbertTokenizer

        albert_dir = kw.get("albert_dir", os.path.join(DOWNLOAD_DIR, "pretrained_lm", "albert"))
        special_tokens = kw.get("special_tokens", list())

        tokenizer = AlbertTokenizer(special_tokens=special_tokens, albert_dir=albert_dir)
        self.special_token_ids = {
            token: tokenizer.get_special_token_id(token)
            for token in tokenizer.special_tokens
        }

        self.model = AlbertModel.from_pretrained("albert-base-v2", cache_dir=albert_dir)

        self.init_embeddings(
            num_words=max(
                max(self.special_token_ids.values()) + 1,
                self.model.embeddings.word_embeddings.num_embeddings
            ),
            num_segments=kw.get(
                "num_segments",
                2
            )
        )

        self.set_finetune("full")

    def set_finetune(self, finetune):
        if finetune == "last":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.encoder.albert_layer_groups[-1].parameters():
                param.requires_grad = True
        else:
            super(AlbertEncoder, self).set_finetune(finetune)

    def forward(
        self,
        x,
        mask=None,
        sent_ids=None,
        type_ids=None,
        pos_ids=None,
        stance_logit=None,
        disco_logit=None,
        mems=None,
        mems_mask=None
    ):
        bsz, x_len = x.size()
        device = x.device

        if mask is None:
            mask = th.ones_like(x)
        if sent_ids is None:
            sent_ids = th.zeros_like(x)
        elif sent_ids.size() != x.size():
            sent_ids = sent_ids.unsqueeze(0)
        if type_ids is None:
            type_ids = th.zeros_like(x)
        elif type_ids.size() != x.size():
            type_ids = type_ids.unsqueeze(0)
        if mems is None:
            mlen = 0
            mems = [None] * self.model.config.num_hidden_layers
            extended_mask = self.model.get_extended_attention_mask(mask, (bsz, x_len), device)
        else:
            mlen = mems[0].size(1)
            if mems_mask is None:
                mems_mask = th.ones_like(mems[0]).to(mask)
            extended_mask = th.cat([mems_mask, mask], dim=1)
            extended_mask = self.model.get_extended_attention_mask(extended_mask, (bsz, mlen+x_len), device)

        word_emb = self.model.embeddings(input_ids=x, position_ids=pos_ids, token_type_ids=type_ids)

        dummy_stance_logit = th.zeros((self.stance_w2.size(0), ), device=self.stance_w2.device, dtype=self.stance_w2.dtype)
        dummy_stance_logit[0].fill_(INF)
        if stance_logit is None:
            stance_logit = dummy_stance_logit.view(1, 1, -1).expand(bsz, x_len, -1)
        stance_weight = F.softmax(stance_logit / self.stance_t, dim=-1)
        stance_emb1 = th.matmul(stance_weight, self.stance_w1)
        dummy_stance_emb1 = self.stance_w1[0]
        refined_stance_emb1 = refine_emb(sent_ids, stance_emb1, dummy_stance_emb1)
        stance_emb2 = th.matmul(stance_weight, self.stance_w2)

        dummy_disco_logit = th.zeros((self.disco_w2.size(0), ), device=self.disco_w2.device, dtype=self.disco_w2.dtype)
        dummy_disco_logit[0].fill_(INF)
        if disco_logit is None:
            disco_logit = dummy_disco_logit.view(1, 1, -1).expand(bsz, x_len, -1)
        disco_weight = F.softmax(disco_logit / self.disco_t, dim=-1)
        disco_emb1 = th.matmul(disco_weight, self.disco_w1)
        dummy_disco_emb1 = self.disco_w1[0]
        refined_disco_emb1 = refine_emb(sent_ids, disco_emb1, dummy_disco_emb1)
        disco_emb2 = th.matmul(disco_weight, self.disco_w2)

        disco_stance_emb = refined_stance_emb1 + stance_emb2 + refined_disco_emb1 + disco_emb2
        hidden = word_emb + disco_stance_emb

        if mask.dim() == 2:
            zero_mask = (mask == 0).unsqueeze(-1)
        else:
            zero_mask = (th.diagonal(mask, dim1=1, dim2=2) == 0).unsqueeze(-1)
        hidden.masked_fill_(zero_mask, 0.0)

        new_mems = []
        # Number of layers in a hidden group
        layers_per_group = int(self.model.config.num_hidden_layers / self.model.config.num_hidden_groups)
        for i in range(self.model.config.num_hidden_layers):
            if self.mem_len > 0:
                new_mems.append(self.cache_mem(hidden, mems[i]))
            else:
                new_mems.append(None)

            # Index of the hidden group
            group_idx = int(i / layers_per_group)

            if mems[i] is None:
                extended_hidden = hidden
                hidden = self.albert_layer_groups[group_idx](hidden, extended_mask)[0]
            else:
                extended_hidden = th.cat([mems[i], hidden], dim=1)
                hidden = self.albert_layer_groups[group_idx](extended_hidden, extended_mask)[0][:, -x_len:]
            hidden.masked_fill_(zero_mask, 0.0)

        if hidden.is_contiguous():
            output = hidden
        else:
            output = hidden.contiguous()
        new_mems = tuple(new_mems)

        return output, new_mems


class XLNetEncoder(TransformerEncoder):
    def __init__(self, **kw):
        super(XLNetEncoder, self).__init__(**kw)

        from transformers import XLNetModel
        from tokenizer import XLNetTokenizer

        xlnet_dir = kw.get("xlnet_dir", os.path.join(DOWNLOAD_DIR, "pretrained_lm", "xlnet"))
        special_tokens = kw.get("special_tokens", list())

        tokenizer = XLNetTokenizer(special_tokens=special_tokens, xlnet_dir=xlnet_dir)
        self.special_token_ids = {
            token: tokenizer.get_special_token_id(token)
            for token in tokenizer.special_tokens
        }

        mem_len = kw.get("mem_len", kw.get("max_len", 1024))
        self.model = XLNetModel.from_pretrained("xlnet-base-cased", cache_dir=xlnet_dir, mem_len=mem_len)

        self.init_embeddings(
            num_words=max(
                max(self.special_token_ids.values()) + 1,
                self.model.word_embedding.num_embeddings
            )
        )

        self.set_finetune("full")

    def init_embeddings(self, num_words, num_segments=1, **kw):
        super(TransformerEncoder, self).init_embeddings(num_words, num_segments, **kw)

        embedding_dim = self.model.word_embedding.embedding_dim
        num_word_embeddings = self.model.word_embedding.num_embeddings

        with th.no_grad():
            # word embeddings
            new_word_embeddings = nn.Embedding(
                num_words, embedding_dim,
                padding_idx=self.model.word_embedding.padding_idx
            )
            new_word_embeddings.weight.data[:num_word_embeddings].copy_(
                self.model.word_embedding.weight
            )
            new_word_embeddings.weight.data[num_word_embeddings:].copy_(
                self.model.word_embedding.weight[self.get_special_token_id(UNK)].unsqueeze(0)
            )

            del self.model.word_embedding
            self.model.word_embedding = new_word_embeddings

    def forward(
        self,
        x,
        mask=None,
        sent_ids=None,
        type_ids=None,
        pos_ids=None,
        stance_logit=None,
        disco_logit=None,
        mems=None,
        mems_mask=None
    ):
        bsz, x_len = x.size()

        if sent_ids is None:
            sent_ids = th.zeros_like(x)
        elif sent_ids.size() != x.size():
            sent_ids = sent_ids.unsqueeze(0)
        if type_ids is None:
            type_ids = th.zeros_like(x)
        elif type_ids.size() != x.size():
            type_ids = type_ids.unsqueeze(0)

        if mems is None:
            mlen = 0
            mems = [None] * self.model.config.num_hidden_layers
        else:
            mlen = mems[0].size(1) if mems[0] is not None else 0
            mems = [m.transpose(0, 1) if m is not None else None for m in mems]
        klen = mlen + x_len
        dtype_float = self.model.dtype
        device = x.device

        if mask is not None:
            if mask.dtype != dtype_float:
                mask = mask.to(dtype_float)
            if mask.dim() == 3:
                attn_mask = (1.0 - mask.permute(1, 2, 0)).contiguous() # qk_len, v_len, bsz
            else:
                attn_mask = (1.0 - mask.transpose(0, 1)).contiguous().unsqueeze(0)
            if mlen > 0:
                if mems_mask is None:
                    mems_mask = th.zeros([attn_mask.size(0), mlen, bsz], dtype=dtype_float, device=device)
                else:
                    if mems_mask.dtype != dtype_float:
                        mems_mask = mems_mask.to(dtype_float)
                    mems_mask = (1.0 - mems_mask.transpose(0, 1))
                    mems_mask = mems_mask.unsqueeze(0).expand(attn_mask.size(0), -1, -1)
                attn_mask = th.cat([mems_mask, attn_mask], dim=1) # qk_len, mv_len, bsz
            attn_mask = attn_mask[:, :, :, None] # qk_len, mv_len, bsz, 1
            non_tgt_mask = -th.eye(x_len).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = th.cat([th.zeros([x_len, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(dtype_float)
        else:
            attn_mask = None
            non_tgt_mask = None

        input_ids = x.transpose(0, 1).contiguous()
        word_emb = self.model.word_embedding(input_ids)
        word_emb = self.model.dropout(word_emb)

        # Segment encoding
        token_type_ids = type_ids.transpose(0, 1).contiguous()
        if mlen > 0:
            mem_pad = th.zeros([mlen, bsz], dtype=th.long, device=device)
            cat_ids = th.cat([mem_pad, token_type_ids], dim=0)
        else:
            cat_ids = token_type_ids
        seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
        seg_mat = F.one_hot(seg_mat, num_classes=2).to(dtype_float)

        dummy_stance_logit = th.zeros((self.stance_w2.size(0), ), device=self.stance_w2.device, dtype=self.stance_w2.dtype)
        dummy_stance_logit[0].fill_(INF)
        if stance_logit is None:
            stance_logit = dummy_stance_logit.view(1, 1, -1).expand(bsz, x_len, -1)
        stance_weight = F.softmax(stance_logit / self.stance_t, dim=-1)
        stance_emb1 = th.matmul(stance_weight, self.stance_w1)
        dummy_stance_emb1 = self.stance_w1[0]
        refined_stance_emb1 = refine_emb(sent_ids, stance_emb1, dummy_stance_emb1)
        stance_emb2 = th.matmul(stance_weight, self.stance_w2)

        dummy_disco_logit = th.zeros((self.disco_w2.size(0), ), device=self.disco_w2.device, dtype=self.disco_w2.dtype)
        dummy_disco_logit[0].fill_(INF)
        if disco_logit is None:
            disco_logit = dummy_disco_logit.view(1, 1, -1).expand(bsz, x_len, -1)
        disco_weight = F.softmax(disco_logit / self.disco_t, dim=-1)
        disco_emb1 = th.matmul(disco_weight, self.disco_w1)
        dummy_disco_emb1 = self.disco_w1[0]
        refined_disco_emb1 = refine_emb(sent_ids, disco_emb1, dummy_disco_emb1)
        disco_emb2 = th.matmul(disco_weight, self.disco_w2)

        disco_stance_emb = refined_stance_emb1 + stance_emb2 + refined_disco_emb1 + disco_emb2
        hidden = word_emb + disco_stance_emb.transpose(0, 1)

        if mask.dim() == 2:
            zero_mask = (mask == 0).transpose(0, 1).unsqueeze(-1)
        else:
            zero_mask = (th.diagonal(mask, dim1=1, dim2=2) == 0).transpose(0, 1).unsqueeze(-1)
        hidden.masked_fill_(zero_mask, 0.0)

        # Positional encoding
        pos_emb = self.model.relative_positional_encoding(x_len, klen, bsz=bsz)
        pos_emb = self.model.dropout(pos_emb)

        new_mems = []

        for i, layer_module in enumerate(self.model.layer):
            if self.mem_len is None:
                new_mems.append(None)
            else:
                new_mems.append(self.model.cache_mem(hidden, mems[i]))

            hidden = layer_module(
                hidden,
                output_g=None,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                seg_mat=seg_mat,
                mems=mems[i],
                target_mapping=None,
                head_mask=None,
                output_attentions=False,
            )[0]
            hidden.masked_fill_(zero_mask, 0.0)

        output = self.model.dropout(hidden)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = output.transpose(1, 0).contiguous()
        if self.mem_len is None:
            new_mems = tuple(new_mems)
        else:
            new_mems = tuple([m.transpose(0, 1).contiguous() for m in new_mems])

        return output, new_mems
