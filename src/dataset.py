import copy
import math
import numpy as np
import json
import torch as th
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
from util import batch_convert_tensor_to_tensor, clean_sentence_for_parsing, batch_convert_len_to_mask


INF = 1e30
_INF = -1e30
PAD = "<pad>"
UNK = "<unk>"
CLS = "<cls>"
SEP = "<sep>"
BOS = "<s>"
EOS = "</s>"
DUMMY = "<dummy>"
CONTEXT = "<context>"
TEXT = "<text>"
STANCES = ["<null>", "<pro>", "<con>"]
STANCE2ID = {x: i for i, x in enumerate(STANCES)}
DISCOS = [
    "<Null>",
    "<Concession>", "<Contrast>",
    "<Reason>", "<Result>", "<Condition>",
    "<Alternative>", "<ChosenAlternative>", "<Conjunction>", "<Exception>", "<Instantiation>", "<Restatement>",
    "<Precedence>", "<Succession>", "<Synchrony>"
]
DISCO2ID = {x: i for i, x in enumerate(DISCOS)}

argument_impact_label_map_3 = {"NOT IMPACTFUL": 0, "MEDIUM IMPACT": 1, "IMPACTFUL": 2}


class BucketSampler(data.Sampler):

    def __init__(self, dataset, group_by, batch_size, shuffle=False, drop_last=False):
        super(BucketSampler, self).__init__(dataset)
        if isinstance(group_by, str):
            group_by = [group_by]
        self.group_by = group_by
        self.cache = OrderedDict()
        for attr in group_by:
            self.cache[attr] = np.array([x[attr] for x in dataset], dtype=np.float32)
        self.data_size = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def make_array(self):
        rand = np.random.rand(self.data_size).astype(np.float32)
        array = np.stack(list(self.cache.values()) + [rand], axis=-1)
        array = array.view(list(zip(list(self.cache.keys()) + ["rand"], [np.float32] * (len(self.cache) + 1)))).flatten()

        return array

    def handle_singleton(self, batches):
        if not self.drop_last and len(batches) > 1 and len(batches[-1]) < self.batch_size // 2:
            merged_batch = np.concatenate([batches[-2], batches[-1]], axis=0)
            batches.pop()
            batches.pop()
            batches.append(merged_batch[:len(merged_batch)//2])
            batches.append(merged_batch[len(merged_batch)//2:])

        return batches

    def __iter__(self):
        array = self.make_array()
        indices = np.argsort(array, axis=0, order=self.group_by)
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        batches = self.handle_singleton(batches)

        if self.shuffle:
            np.random.shuffle(batches)

        batch_idx = 0
        while batch_idx < len(batches) - 1:
            yield batches[batch_idx]
            batch_idx += 1
        if len(batches) > 0 and (len(batches[batch_idx]) == self.batch_size or not self.drop_last):
            yield batches[batch_idx]

    def __len__(self):
        if self.drop_last:
            return math.floor(self.data_size / self.batch_size)
        else:
            return math.ceil(self.data_size / self.batch_size)

class DiscoChainDataset(data.Dataset):
    def __init__(self, data=None, **kw):
        super(DiscoChainDataset, self).__init__()
        if data is not None:
            self.data = data
        else:
            self.data = list()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def process_text(text, max_num_text, dummy_text, pad_id, pre_pad=True):
        bsz = len(text)
        padded_text = []
        if max_num_text > 0:
            for i in range(bsz):
                nt = min(len(text[i]), max_num_text)
                padded_text.append([dummy_text.clone() for j in range(max_num_text - nt)] + text[i][-nt:])

            text_matrix = []
            text_lens = []
            for j in range(max_num_text):
                text_len = max([padded_text[i][j].size(0) for i in range(bsz)])
                text_matrix.append(
                    batch_convert_tensor_to_tensor(
                        [padded_text[i][j] for i in range(bsz)],
                        max_seq_len=text_len,
                        pad_id=pad_id,
                        pre_pad=pre_pad
                    )
                )
                text_lens.append(text_len)
            text_matrix = th.cat(text_matrix, dim=1)
            text_mask_matrix = (text_matrix != pad_id)
            text_sent_ids_matrix = th.cat(
                [
                    th.empty((bsz, text_lens[j]), dtype=th.long).fill_(j)
                    for j in range(max_num_text)
                ],
                dim=1
            )
        else:
            text_matrix = dummy_text.clone().unsqueeze(0).repeat(bsz, 1)
            text_mask_matrix = (dummy_text != pad_id).unsqueeze(0).repeat(bsz, 1)
            text_sent_ids_matrix = th.zeros_like(text_matrix)

        return text_matrix, text_mask_matrix, text_sent_ids_matrix

    @staticmethod
    def batchify(batch, label_map, tokenizer, max_num_text=1, max_num_context=1, dummy_text=PAD, pre_pad=True):
        assert max_num_text > 0
        assert max_num_context >= 0

        num_labels = max(label_map.values()) + 1
        dummy_text = tokenizer.encode(dummy_text)
        pad_id = tokenizer.get_special_token_id(PAD)

        valid_batch, _id, label, labels = list(), list(), list(), list()
        for i, x in enumerate(batch):
            if "label" in x:
                idx = label_map.get(x["label"])
                if idx != -1:
                    valid_batch.append(x)
                    _id.append(x["id"])
                    label.append(idx)
                    vec = th.zeros((num_labels, ), dtype=th.float)
                    vec[idx] = 1
                    labels.append(vec)
            elif "labels" in x:
                ind = list()
                for lb in x["labels"]:
                    idx = label_map.get(lb, -1)
                    if idx != -1:
                        ind.append(idx)
                if len(ind) > 0:
                    valid_batch.append(x)
                    _id.append(x["id"])
                    label.append(ind[0])
                    vec = th.zeros((num_labels, ), dtype=th.float)
                    vec[ind] = 1
                    labels.append(vec)

        label = th.tensor(label, dtype=th.long)
        labels = th.stack(labels, dim=0)

        max_num_context = min(max_num_context, max([len(x["context"]) for x in valid_batch]))
        context, context_mask, context_sent_ids = DiscoChainDataset.process_text(
            [x["context"] for x in valid_batch], max_num_context, dummy_text, pad_id=pad_id, pre_pad=pre_pad
        )

        max_num_text = min(max_num_text, max([len(x["text"]) for x in valid_batch]))
        text, text_mask, text_sent_ids = DiscoChainDataset.process_text(
            [x["text"] for x in valid_batch], max_num_text, dummy_text, pad_id=pad_id, pre_pad=pre_pad
        )

        dummy_stance = th.zeros((1, len(STANCES)))
        dummy_stance[0][0] = 1.0
        dummy_disco = th.zeros((1, len(DISCOS)))
        dummy_disco[0][0] = 1.0
        max_num = max_num_context + max_num_text
        stance_logit = []
        disco_logit = []
        for i, x in enumerate(valid_batch):
            index = th.cat([context_sent_ids[i], text_sent_ids[i] + (context_sent_ids[i][-1] + 1)])
            stance_logit.append(
                th.index_select(
                    th.cat([dummy_stance.repeat(max_num - x["stance_logit"].size(0), 1), x["stance_logit"]], dim=0),
                    dim=0,
                    index=index
                )
            )
            disco_logit.append(
                th.index_select(
                    th.cat([dummy_disco.repeat(max_num - x["discourse_logit"].size(0), 1), x["discourse_logit"]], dim=0),
                    dim=0,
                    index=index
                )
            )
        stance_logit = th.stack(stance_logit, dim=0)
        disco_logit = th.stack(disco_logit, dim=0)

        return _id, context, context_mask, context_sent_ids, text, text_mask, text_sent_ids, \
            stance_logit, disco_logit, labels, label

    def load_jsonl(self, jsonl_file_path, tokenizer, max_len=-1):
        stance_onehot = th.eye(len(STANCES)) * INF
        disco_onehot = th.eye(len(DISCOS)) * INF
        with open(jsonl_file_path, "r") as f:
            self.data = list()
            for line in tqdm(f):
                x = json.loads(line)
                if "text" in x:
                    x["text"] = [tokenizer.encode(clean_sentence_for_parsing(t), max_len) for t in x["text"]]
                if "context" in x:
                    x["context"] = [tokenizer.encode(clean_sentence_for_parsing(c), max_len) for c in x["context"]]
                if "stance_label" not in x:
                    x["stance_label"] = [STANCES[0]] * (len(x["context"]) + len(x["text"]))
                    x["stance_logit"] = stance_onehot[0].unsqueeze(0).repeat(len(x["context"]) + len(x["text"]), 1)
                else:
                    if "stance_logit" not in x:
                        x["stance_logit"] = th.index_select(
                            stance_onehot,
                            dim=0,
                            index=th.tensor([STANCE2ID[lb] for lb in x["stance_label"]], dtype=th.long)
                        )
                    else:
                        x["stance_logit"] = th.tensor(x["stance_logit"])
                if "discourse_label" not in x:
                    x["discourse_label"] = [DISCOS[0]] * (len(x["context"]) + len(x["text"]))
                    x["discourse_logit"] = disco_onehot[0].unsqueeze(0).repeat(len(x["context"]) + len(x["text"]), 1)
                else:
                    if "discourse_logit" not in x:
                        x["discourse_logit"] = th.index_select(
                            disco_onehot,
                            dim=0,
                            index=th.tensor([DISCO2ID[lb] for lb in x["discourse_label"]], dtype=th.long)
                        )
                    else:
                        x["discourse_logit"] = th.tensor(x["discourse_logit"])
                self.data.append(x)
        return self


class DiscoDataset(data.Dataset):
    def __init__(self, data=None, **kw):
        super(DiscoDataset, self).__init__()

        if data is not None:
            self.data = data
        else:
            self.data = list()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def batchify(batch, label_map, pre_pad=True):
        num_labels = max(label_map.values()) + 1

        valid_batch, _id, label, labels = list(), list(), list(), list()
        for i, x in enumerate(batch):
            if "label" in x:
                idx = label_map.get(x["label"])
                if idx != -1:
                    valid_batch.append(x)
                    _id.append(x["id"])
                    label.append(idx)
                    vec = th.zeros((num_labels, ), dtype=th.float)
                    vec[idx] = 1
                    labels.append(vec)
            elif "labels" in x:
                ind = list()
                for lb in x["labels"]:
                    idx = label_map.get(lb, -1)
                    if idx != -1:
                        ind.append(idx)
                if len(ind) > 0:
                    valid_batch.append(x)
                    _id.append(x["id"])
                    label.append(ind[0])
                    vec = th.zeros((num_labels, ), dtype=th.float)
                    vec[ind] = 1
                    labels.append(vec)

        label = th.tensor(label, dtype=th.long)
        labels = th.stack(labels, dim=0)

        disco = batch_convert_tensor_to_tensor(
            [x["discourse"] for x in valid_batch], max_seq_len=-1, pad_id=0.0, pre_pad=pre_pad
        )
        mask = batch_convert_len_to_mask([len(x["discourse"]) for x in valid_batch], max_seq_len=-1, pre_pad=pre_pad)

        return _id, disco, mask, labels, label

    def load_jsonl(self, jsonl_file_path):
        with open(jsonl_file_path, "r") as f:
            self.data = list()
            for line in tqdm(f):
                x = json.loads(line)
                if "discourse" in x:
                    x["discourse"] = th.tensor(x["discourse"])
                self.data.append(x)
        return self
