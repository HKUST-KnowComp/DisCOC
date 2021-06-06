import os
import torch as th
from dataset import BOS, CLS, EOS, PAD, SEP, UNK


DOWNLOAD_DIR = os.path.dirname(os.path.abspath(__file__))


class Tokenizer:
    def __init__(self, **kw):
        self._tokenizer = None
        if "special_tokens" in kw:
            self.special_tokens = set(kw["special_tokens"])
        else:
            self.special_tokens = set()

    def tokenize(self, sentence, max_len=-1):
        """
        :param sentence: a string of sentence
        :param max_len: maximum length
        :return encoded_result: a list of tokens
        """
        raise NotImplementedError

    def encode(self, sentence, max_len=-1, return_tensors=True):
        """
        :param sentence: a string of sentence
        :param max_len: maximum length
        :return encoded_result: a list of ids or a tensor if return_tensors=True
        """
        raise NotImplementedError

    def get_special_token(self, key):
        raise NotImplementedError

    def get_special_token_id(self, key):
        raise NotImplementedError


class SpacyTokenizer(Tokenizer):
    def __init__(self, **kw):
        super(SpacyTokenizer, self).__init__(**kw)
        from spacy.attrs import ORTH
        from spacy.lang.en import English

        self.special_tokens.add(PAD)
        self.special_tokens.add(UNK)
        self.special_tokens.add(BOS)
        self.special_tokens.add(EOS)

        _nlp = English()
        self._tokenizer = _nlp.Defaults.create_tokenizer(_nlp)
        for t in self.special_tokens:
            self._tokenizer.add_special_case(t, [{ORTH: t}])

        self.word2idx = dict(
            zip(self.special_tokens, range(len(self.special_tokens)))
        )

        if "word2vec_file" in kw:
            with open(kw["word2vec_file"], "r", encoding="utf-8") as f:
                line = f.readline()
                line = line.split(" ", 1)
                try:
                    # word, dim
                    if float(line[0]) and float(line[1]):
                        pass
                except BaseException as e:
                    if isinstance(e, ValueError):
                        word = line[0]
                        if word not in self.word2idx:
                            self.word2idx[word] = len(self.word2idx)
                finally:
                    pass
                for line in f:
                    word = line.split(" ", 1)[0]
                    if word not in self.word2idx:
                        self.word2idx[word] = len(self.word2idx)
        elif "vocab" in kw:
            for word in kw["vocab"]:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
        elif "word2idx" in kw:
            self.word2idx.update(kw["word2idx"])

        self.special_tokens.add(CLS)
        self.special_tokens.add(SEP)

    def tokenize(self, sentence, max_len=-1):
        assert isinstance(sentence, str)

        tokens = self._tokenizer(sentence)

        if max_len > 0 and len(tokens) > max_len - 2:
            tokens = tokens[:(max_len - 2)]

        return [self.get_special_token(BOS)] + [t.text for t in tokens] + [self.get_special_token(EOS)]

    def encode(self, sentence, max_len=-1, return_tensors=True):
        assert isinstance(sentence, str)

        tokens = self.tokenize(sentence, max_len)

        ids = []
        for t in tokens:
            if t in self.word2idx:
                ids.append(self.word2idx[t])
                continue
            tl = t.lower()
            if tl in self.word2idx:
                ids.append(self.word2idx[tl])
            else:
                ids.append(self.get_special_token_id(UNK))

        if return_tensors:
            ids = th.tensor(ids, dtype=th.long)
        return ids

    def get_special_token(self, key):
        if key == PAD:
            return PAD
        if key == UNK:
            return UNK
        if key == BOS or key == CLS:
            return BOS
        if key == EOS or key == SEP:
            return EOS
        if key in self.special_tokens:
            return key
        raise ValueError("Error: %s is not the special token." % (key))

    def get_special_token_id(self, key):
        return self.word2idx[self.get_special_token(key)]


class TransformerTokenizer(Tokenizer):
    def __init__(self, **kw):
        super(TransformerTokenizer, self).__init__(**kw)

        self._tokenizer = None

    def tokenize(self, sentence, max_len=-1):
        assert isinstance(sentence, str)

        tokens = [self.get_special_token(CLS)]
        tokens.extend(self._tokenizer.tokenize(sentence))

        if max_len > 0 and len(tokens) > max_len - 1:
            tokens = tokens[:(max_len - 1)]

        tokens.append(self.get_special_token(SEP))

        return tokens

    def encode(self, sentence, max_len=-1, return_tensors=True):
        assert isinstance(sentence, str)

        tokens = self.tokenize(sentence, max_len)
        ids = self._tokenizer.convert_tokens_to_ids(tokens)

        if return_tensors:
            ids = th.tensor(ids)
        return ids

    def get_special_token(self, key):
        if key == PAD:
            return self._tokenizer.pad_token
        if key == UNK:
            return self._tokenizer.unk_token
        if key == CLS or key == BOS:
            return self._tokenizer.cls_token
        if key == SEP or key == EOS:
            return self._tokenizer.sep_token
        if key in self.special_tokens:
            return key
        raise ValueError("Error: %s is not the special token." % (key))

    def get_special_token_id(self, key):
        return self._tokenizer.convert_tokens_to_ids(
            self.get_special_token(key)
        )


class BertTokenizer(TransformerTokenizer):
    def __init__(self, **kw):
        super(BertTokenizer, self).__init__(**kw)
        from transformers import BertTokenizerFast as _BertTokenizer

        bert_dir = kw.get(
            "bert_dir", os.path.join(DOWNLOAD_DIR, "pretrained_lm", "bert")
        )

        self._tokenizer = _BertTokenizer.from_pretrained(
            "bert-base-uncased", cache_dir=bert_dir
        )

        for t in self.special_tokens:
            self._tokenizer.add_tokens(t, special_tokens=True)

        self.special_tokens.add(self._tokenizer.pad_token)
        self.special_tokens.add(self._tokenizer.unk_token)
        self.special_tokens.add(self._tokenizer.cls_token)
        self.special_tokens.add(self._tokenizer.sep_token)
        self.special_tokens.add(PAD)
        self.special_tokens.add(UNK)
        self.special_tokens.add(BOS)
        self.special_tokens.add(EOS)
        self.special_tokens.add(CLS)
        self.special_tokens.add(SEP)


class RobertaTokenizer(TransformerTokenizer):
    def __init__(self, **kw):
        super(RobertaTokenizer, self).__init__(**kw)
        from transformers import RobertaTokenizerFast as _RobertaTokenizer

        roberta_dir = kw.get(
            "roberta_dir", os.path.join(DOWNLOAD_DIR, "pretrained_lm", "roberta")
        )

        self._tokenizer = _RobertaTokenizer.from_pretrained(
            "roberta-base", cache_dir=roberta_dir
        )

        for t in self.special_tokens:
            self._tokenizer.add_tokens(t, special_tokens=True)

        self.special_tokens.add(self._tokenizer.pad_token)
        self.special_tokens.add(self._tokenizer.unk_token)
        self.special_tokens.add(self._tokenizer.cls_token)
        self.special_tokens.add(self._tokenizer.sep_token)
        self.special_tokens.add(PAD)
        self.special_tokens.add(UNK)
        self.special_tokens.add(BOS)
        self.special_tokens.add(EOS)
        self.special_tokens.add(CLS)
        self.special_tokens.add(SEP)

class AlbertTokenizer(TransformerTokenizer):
    def __init__(self, **kw):
        super(AlbertTokenizer, self).__init__(**kw)
        from transformers import AlbertTokenizer as _AlbertTokenizer

        albert_dir = kw.get(
            "albert_dir", os.path.join(DOWNLOAD_DIR, "pretrained_lm", "albert")
        )

        self._tokenizer = _AlbertTokenizer.from_pretrained(
            "albert-base-v2", cache_dir=albert_dir
        )

        for t in self.special_tokens:
            self._tokenizer.add_tokens(t, special_tokens=True)

        self.special_tokens.add(self._tokenizer.pad_token)
        self.special_tokens.add(self._tokenizer.unk_token)
        self.special_tokens.add(self._tokenizer.cls_token)
        self.special_tokens.add(self._tokenizer.sep_token)
        self.special_tokens.add(PAD)
        self.special_tokens.add(UNK)
        self.special_tokens.add(BOS)
        self.special_tokens.add(EOS)
        self.special_tokens.add(CLS)
        self.special_tokens.add(SEP)

class XLNetTokenizer(TransformerTokenizer):
    def __init__(self, **kw):
        super(XLNetTokenizer, self).__init__(**kw)
        from transformers import XLNetTokenizer as _XLNetTokenizer

        xlnet_dir = kw.get(
            "xlnet_dir", os.path.join(DOWNLOAD_DIR, "pretrained_lm", "xlnet")
        )

        self._tokenizer = _XLNetTokenizer.from_pretrained(
            "xlnet-base-cased", cache_dir=xlnet_dir
        )

        for t in self.special_tokens:
            self._tokenizer.add_tokens(t, special_tokens=True)

        self.special_tokens.add(self._tokenizer.pad_token)
        self.special_tokens.add(self._tokenizer.unk_token)
        self.special_tokens.add(self._tokenizer.cls_token)
        self.special_tokens.add(self._tokenizer.sep_token)
        self.special_tokens.add(PAD)
        self.special_tokens.add(UNK)
        self.special_tokens.add(BOS)
        self.special_tokens.add(EOS)
        self.special_tokens.add(CLS)
        self.special_tokens.add(SEP)

    def tokenize(self, sentence, max_len=-1):
        assert isinstance(sentence, str)

        tokens = self._tokenizer.tokenize(sentence)

        if max_len > 0 and len(tokens) > max_len - 2:
            tokens = tokens[:(max_len - 2)]

        tokens.append(self.get_special_token(SEP))
        tokens.append(self.get_special_token(CLS))

        return tokens
