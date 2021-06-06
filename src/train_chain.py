import argparse
import copy
import datetime
import gc
import logging
import math
import os
import pprint
from collections import OrderedDict
from functools import partial

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.optim import Adam

try:
    from torch.utils.tensorboard import SummaryWriter
except BaseException as e:
    from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from classifier import *
from dataset import *
from log import *
from optim import *
from tokenizer import *
from util import *

INF = 1e30
_INF = -1e30
BUCKET_DIFF = 2


def preprocess_batch(batch, cls_token_id, sep_token_id, context_token_id, text_token_id):
    _id, context, context_mask, context_sent_ids, text, text_mask, text_sent_ids, \
        stance_logit, disco_logit, labels, label = batch

    if args.encoder == "xlnet":
        # ... <sep> <cls> --> ... <sep> <context>
        context_cls = context == cls_token_id
        context.masked_fill_(context_cls, context_token_id)

        # ... <sep> <cls> --> ... <sep> <text>
        text_cls = text == cls_token_id
        text.masked_fill_(text_cls, text_token_id)

        # ... <sep> <context> | ... <sep> <context> | ... <sep> <text> | ... <sep> <cls>
        text.index_fill_(dim=1, index=text_cls.cumsum(dim=1).max(dim=1)[1], value=cls_token_id)

    else:
        if args.mode == "discoc" or args.cls_last:
            # <cls> ... <sep> --> <context> ... <context>
            context_cls = context == cls_token_id
            context_sep = context == sep_token_id
            context.masked_fill_(context_cls, context_token_id)
            context.masked_fill_(context_sep, context_token_id)

            # <cls> ... <sep> --> <text> ... <text>
            text_cls = text == cls_token_id
            text_sep = text == sep_token_id
            text.masked_fill_(text_cls, text_token_id)
            text.masked_fill_(text_sep, text_token_id)
            text.masked_fill_(text == cls_token_id, text_token_id)

            # <sep> ... <context> | <sep> ... <context> | <sep> ... <text> | <sep> ... <cls>
            context.masked_fill_(context_cls, sep_token_id)
            text.masked_fill_(text_cls, sep_token_id)
            text.index_fill_(dim=1, index=text_sep.cumsum(dim=1).max(dim=1)[1], value=cls_token_id)
        else:
            # <cls> ... <sep> --> <context> ... <sep>
            context_cls = context == cls_token_id
            context.masked_fill_(context_cls, context_token_id)

            # <cls> ... <sep> --> <text> ... <sep>
            text_cls = text == cls_token_id
            text.masked_fill_(text_cls, text_token_id)

            # <context> ... <sep> | <context> ... <sep> | <cls> ... <sep> | <text> ... <sep>
            text.index_fill_(dim=1, index=text_cls.max(dim=1)[1], value=cls_token_id)

    return _id, context, context_mask, context_sent_ids, text, text_mask, text_sent_ids, \
        stance_logit, disco_logit, labels, label


def eval_epoch(args, logger, writer, model, data_type, data_loader, device, epoch):
    context_token_id = model.encoder.get_special_token_id(CONTEXT)
    text_token_id = model.encoder.get_special_token_id(TEXT)
    sep_token_id = model.encoder.get_special_token_id(SEP)
    cls_token_id = model.encoder.get_special_token_id(CLS)

    model.eval()
    epoch_steps = len(data_loader)
    total_steps = args.epochs * epoch_steps
    total_cnt = 0
    total_ce = 0.0

    results = {
        "data": {
            "id": list(),
            "label": list(),
            "labels": list(),
            "stance_logit": list(),
            "disco_logit": list(),
        },
        "prediction": {
            "prob": list(),
            "pred": list()
        },
        "error": {
            "ce": list(),
            "mean_ce": INF,
        },
        "evaluation": {
            "accuracy": dict(),
            "precision_recall_f1": dict()
        },
        "model": {
            "stance_tau": list(),
            "disco_tau": list()
        }
    }

    with th.no_grad():
        for batch_id, batch in enumerate(data_loader):
            step = epoch * epoch_steps + batch_id
            _id, context, context_mask, context_sent_ids, text, text_mask, text_sent_ids, \
                stance_logit, disco_logit, labels, label = preprocess_batch(
                    batch, cls_token_id, sep_token_id, context_token_id, text_token_id
                )
            bsz = len(_id)

            results["data"]["id"].extend(_id)
            results["data"]["labels"].extend(labels.numpy())
            results["data"]["label"].extend(label.numpy())

            text = text.to(device)
            context = context.to(device)
            if text_mask is not None:
                text_mask = text_mask.to(device)
            if context_mask is not None:
                context_mask = context_mask.to(device)
            text_sent_ids = text_sent_ids.to(device)
            context_sent_ids = context_sent_ids.to(device)
            stance_logit = stance_logit.to(device) if args.add_stance_embed else None
            disco_logit = disco_logit.to(device) if args.add_disco_embed else None
            labels = labels.to(device)
            label = label.to(device)

            output = model(
                context, text, context_mask, text_mask, context_sent_ids, text_sent_ids, stance_logit, disco_logit
            )
            logp = F.log_softmax(output, dim=-1)
            prob = logp.exp()

            ce = F.nll_loss(logp, label, reduction="none")

            prob = prob.cpu()
            results["prediction"]["prob"].extend(prob.numpy())
            results["prediction"]["pred"].extend(prob.argmax(dim=1).numpy())

            results["error"]["ce"].extend(ce.cpu().numpy())

            avg_ce = ce.mean().item()

            total_cnt += bsz
            total_ce += avg_ce * bsz

            stance_tau = model.encoder.stance_t.detach().item()
            disco_tau = model.encoder.disco_t.detach().item()



            if args.loss == "ce":
                avg_loss = avg_ce
            else:
                raise NotImplementedError("Error: loss=%s is not supported now." % (args.loss))

            if writer:
                writer.add_scalar("%s/loss" % (data_type), avg_loss, step)
                writer.add_scalar("%s/ce" % (data_type), avg_ce, step)
                writer.add_scalar("%s/stance_tau" % (data_type), stance_tau, step)
                writer.add_scalar("%s/disco_tau" % (data_type), disco_tau, step)

        epoch_avg_ce = total_ce / total_cnt if total_cnt > 0 else 0.0

        if args.loss == "ce":
            epoch_avg_loss = epoch_avg_ce
        else:
            raise NotImplementedError("Error: loss=%s is not supported now." % (args.loss))

        pred = np.stack(results["prediction"]["pred"], axis=0)
        labels = np.stack(results["data"]["labels"], axis=0)
        label = np.stack(results["data"]["label"], axis=0)

        results["error"]["mean_ce"] = epoch_avg_ce
        results["evaluation"]["accuracy"] = accuracy_score(label, pred)
        results["evaluation"]["precision_recall_f1"] = precision_recall_fscore_support(label, pred, average="macro")[:3]
        results["model"]["stance_tau"].append(stance_tau)
        results["model"]["disco_tau"].append(disco_tau)

        if writer:
            writer.add_scalar("%s/loss-epoch" % (data_type), epoch_avg_loss, epoch)
            writer.add_scalar("%s/ce-epoch" % (data_type), epoch_avg_ce, epoch)
            writer.add_scalar("%s/stance_tau-epoch" % (data_type), stance_tau, step)
            writer.add_scalar("%s/disco_tau-epoch" % (data_type), disco_tau, step)

        if logger:
            logger.info("-" * 80)
            logger.info(
                generate_log_line(
                    data_type + "-best",
                    epochs=epoch,
                    total_epochs=args.epochs,
                    **{
                        "\n" + " " * 26 +
                        "loss-epoch":
                            avg_loss,
                        "ce-epoch":
                            epoch_avg_loss,
                        "stance_tau-epoch":
                            stance_tau,
                        "disco_tau-epoch":
                            disco_tau,
                        "\n" + " " * 26 +
                        "accuracy":
                            pprint.pformat(results["evaluation"]["accuracy"]).replace("\n", "\n" + " " * 26),
                        "\n" + " " * 26 +
                        "precision_recall_f1":
                            pprint.pformat(results["evaluation"]["precision_recall_f1"]).replace("\n", "\n" + " " * 26)
                    },
                )
            )

    gc.collect()
    return epoch_avg_loss, results


def train_epoch(args, logger, writer, model, optimizer, scheduler, data_type, data_loader, device, epoch):
    context_token_id = model.encoder.get_special_token_id(CONTEXT)
    text_token_id = model.encoder.get_special_token_id(TEXT)
    sep_token_id = model.encoder.get_special_token_id(SEP)
    cls_token_id = model.encoder.get_special_token_id(CLS)

    model.train()
    epoch_steps = len(data_loader)
    total_steps = args.epochs * epoch_steps
    total_cnt = 0
    total_ce = 0.0

    results = {
        "data": {
            "id": list(),
            "label": list(),
            "labels": list(),
            "stance_logit": list(),
            "disco_logit": list(),
        },
        "prediction": {
            "prob": list(),
            "pred": list()
        },
        "error": {
            "ce": list(),
            "mean_ce": INF,
        },
        "evaluation": {
            "accuracy": dict(),
            "precision_recall_f1": dict()
        },
        "model": {
            "stance_tau": list(),
            "disco_tau": list()
        }
    }

    for batch_id, batch in enumerate(data_loader):
        step = epoch * epoch_steps + batch_id
        _id, context, context_mask, context_sent_ids, text, text_mask, text_sent_ids, \
            stance_logit, disco_logit, labels, label = preprocess_batch(
                batch, cls_token_id, sep_token_id, context_token_id, text_token_id
            )
        bsz = len(_id)

        results["data"]["id"].extend(_id)
        results["data"]["labels"].extend(labels.numpy())
        results["data"]["label"].extend(label.numpy())

        text = text.to(device)
        context = context.to(device)
        if text_mask is not None:
            text_mask = text_mask.to(device)
        if context_mask is not None:
            context_mask = context_mask.to(device)
        text_sent_ids = text_sent_ids.to(device)
        context_sent_ids = context_sent_ids.to(device)
        stance_logit = stance_logit.to(device) if args.add_stance_embed else None
        disco_logit = disco_logit.to(device) if args.add_disco_embed else None
        labels = labels.to(device)
        label = label.to(device)

        output = model(
            context, text, context_mask, text_mask, context_sent_ids, text_sent_ids, stance_logit, disco_logit
        )
        logp = F.log_softmax(output, dim=-1)
        prob = logp.exp()

        if args.loss == "ce":
            loss = F.nll_loss(logp, label, reduction="none")
        else:
            raise NotImplementedError("Error: loss=%s is not supported now." % (args.loss))

        output = output.detach()
        logp = logp.detach()
        prob = prob.detach()

        ce = F.nll_loss(logp, label, reduction="none")

        prob = prob.cpu()
        results["prediction"]["prob"].extend(prob.numpy())
        results["prediction"]["pred"].extend(prob.argmax(dim=1).numpy())

        results["error"]["ce"].extend(ce.cpu().numpy())

        avg_ce = ce.mean().item()

        total_cnt += bsz
        total_ce += avg_ce * bsz

        stance_tau = model.encoder.stance_t.detach().item()
        disco_tau = model.encoder.disco_t.detach().item()

        model.encoder.stance_t.mul_((1 - 1 / total_steps)) # -> 1/e
        model.encoder.disco_t.mul_((1 - 1 / total_steps)) # -> 1/e

        if args.loss == "ce":
            avg_loss = avg_ce
        else:
            raise NotImplementedError("Error: loss=%s is not supported now." % (args.loss))

        loss.mean().backward()
        if args.max_grad_norm > 0:
            th.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        if writer:
            writer.add_scalar("%s/loss" % (data_type), avg_loss, step)
            writer.add_scalar("%s/ce" % (data_type), avg_ce, step)
            writer.add_scalar("%s/stance_tau" % (data_type), stance_tau, step)
            writer.add_scalar("%s/disco_tau" % (data_type), disco_tau, step)

        if logger and (batch_id == epoch_steps - 1 or batch_id % args.print_every == 0):
            logger.info(
                generate_log_line(
                    data_type,
                    epochs=epoch,
                    total_epochs=args.epochs,
                    step=batch_id,
                    total_steps=epoch_steps,
                    **{
                        "\n" + " " * 26 +
                        "loss":
                            avg_loss,
                        "ce":
                            avg_ce,
                        "stance_tau":
                            stance_tau,
                        "disco_tau":
                            disco_tau,
                        "\n" + " " * 26 +
                        "gold":
                            ", ".join(["{:4.4f}".format(x) for x in results["data"]["labels"][-1]]),
                        "\n" + " " * 26 +
                        "pred":
                            ", ".join(["{:4.4f}".format(x) for x in results["prediction"]["prob"][-1]])
                    },
                )
            )

    epoch_avg_ce = total_ce / total_cnt if total_cnt > 0 else 0.0

    if args.loss == "ce":
        epoch_avg_loss = epoch_avg_ce
    else:
        raise NotImplementedError("Error: loss=%s is not supported now." % (args.loss))

    pred = np.stack(results["prediction"]["pred"], axis=0)
    labels = np.stack(results["data"]["labels"], axis=0)
    label = np.stack(results["data"]["label"], axis=0)

    results["error"]["mean_ce"] = epoch_avg_ce
    results["evaluation"]["accuracy"] = accuracy_score(label, pred)
    results["evaluation"]["precision_recall_f1"] = precision_recall_fscore_support(label, pred, average="macro")[:3]
    results["model"]["stance_tau"].append(stance_tau)
    results["model"]["disco_tau"].append(disco_tau)

    if writer:
        writer.add_scalar("%s/loss-epoch" % (data_type), epoch_avg_loss, epoch)
        writer.add_scalar("%s/ce-epoch" % (data_type), epoch_avg_ce, epoch)
        writer.add_scalar("%s/stance_tau-epoch" % (data_type), stance_tau, step)
        writer.add_scalar("%s/disco_tau-epoch" % (data_type), disco_tau, step)

    if logger:
        logger.info("-" * 80)
        logger.info(
            generate_log_line(
                data_type,
                epochs=epoch,
                total_epochs=args.epochs,
                **{
                    "\n" + " " * 26 +
                    "loss-epoch":
                        avg_loss,
                    "ce-epoch":
                        epoch_avg_loss,
                    "stance_tau-epoch":
                        stance_tau,
                    "disco_tau-epoch":
                        disco_tau,
                    "\n" + " " * 26 +
                    "accuracy":
                        pprint.pformat(results["evaluation"]["accuracy"]).replace("\n", "\n" + " " * 26),
                    "\n" + " " * 26 +
                    "precision_recall_f1":
                        pprint.pformat(results["evaluation"]["precision_recall_f1"]).replace("\n", "\n" + " " * 26)
                },
            )
        )

    gc.collect()
    return epoch_avg_loss, results


def train(args, logger, writer):
    # set device
    if args.gpu_ids is None or len(args.gpu_ids) == 0:
        device = th.device("cpu")
    else:
        if isinstance(args.gpu_ids, int):
            args.gpu_ids = [args.gpu_ids]
        device = th.device("cuda:%d" % args.gpu_ids[0])
        th.cuda.set_device(device)

    # label mapping and number of labels
    if args.label_map == "argument_impact_label_map_3":
        label_map = argument_impact_label_map_3
        num_labels = 3
    else:
        raise NotImplementedError

    # tokenizer
    special_tokens = [CONTEXT, TEXT, DUMMY]
    special_tokens.extend(STANCES)
    special_tokens.extend(DISCOS)
    if args.encoder == "lstm":
        tokenizer = SpacyTokenizer(special_tokens=special_tokens, **vars(args))
    elif args.encoder == "bert":
        tokenizer = BertTokenizer(special_tokens=special_tokens, **vars(args))
    elif args.encoder == "roberta":
        tokenizer = RobertaTokenizer(special_tokens=special_tokens, **vars(args))
    elif args.encoder == "albert":
        tokenizer = AlbertTokenizer(special_tokens=special_tokens, **vars(args))
    elif args.encoder == "xlnet":
        tokenizer = XLNetTokenizer(special_tokens=special_tokens, **vars(args))
    special_token_ids = {t: th.tensor(tokenizer.get_special_token_id(t)).view(1, ) for t in special_tokens}

    # load data
    datasets = OrderedDict(
        {
            "train": DiscoChainDataset().load_jsonl(args.train_dataset_path, tokenizer=tokenizer, max_len=args.max_len),
            "valid": DiscoChainDataset().load_jsonl(args.valid_dataset_path, tokenizer=tokenizer, max_len=args.max_len),
            "test": DiscoChainDataset().load_jsonl(args.test_dataset_path, tokenizer=tokenizer, max_len=args.max_len)
        }
    )

    logger.info("train:valid:test = %d:%d:%d" % (len(datasets["train"]), len(datasets["valid"]), len(datasets["test"])))

    # build data loaders
    data_loaders = OrderedDict()
    batchify = partial(
        DiscoChainDataset.batchify,
        tokenizer=tokenizer,
        label_map=label_map,
        max_num_text=args.max_num_text,
        max_num_context=args.max_num_context,
        dummy_text=DUMMY,
        pre_pad=args.cls_last or args.encoder == "xlnet"
    )

    for data_type in datasets:
        if args.add_stance_token or args.add_disco_token:
            for x in datasets[data_type]:
                len_ctx = len(x["context"])
                for i in range(len(x["stance_label"])):
                    if i < len_ctx:
                        ctx = [x["context"][i][:1]]
                        if args.add_stance_token:
                            ctx.append(special_token_ids[x["stance_label"][i]])
                        if args.add_disco_token:
                            ctx.append(special_token_ids[(x["discourse_label"][i - 1] if i > 0 else "<Null>")])
                        ctx.append(x["context"][i][1:])
                        x["context"][i] = th.cat(ctx)
                    else:
                        tx = [x["text"][i - len_ctx][:1]]
                        if args.add_stance_token:
                            ctx.append(special_token_ids[x["stance_label"][i]])
                        if args.add_disco_token:
                            ctx.append(special_token_ids[(x["discourse_label"][i - 1] if i > 0 else "<Null>")])
                        tx.append(x["text"][i - len_ctx][1:])
                        x["text"][i - len_ctx] = th.cat(tx)
        for x in datasets[data_type]:
            ctx_idx1, ctx_idx2 = max(len(x["context"]) - args.max_num_context, 0), len(x["context"])
            tx_idx1, tx_idx2 = 0, min(len(x["text"]), args.max_num_text)
            x["context"] = x["context"][ctx_idx1:ctx_idx2]
            x["context_len"] = sum([math.ceil(c.size(0) / BUCKET_DIFF) * BUCKET_DIFF for c in x["context"]])
            x["num_context"] = len(x["context"])
            x["text"] = x["text"][tx_idx1:tx_idx2]
            x["text_len"] = sum([math.ceil(t.size(0) / BUCKET_DIFF) * BUCKET_DIFF for t in x["text"]])
            x["max_num_text"] = len(x["text"])
            tx_idx1 += len(x["context"])
            tx_idx2 += len(x["context"])
            x["stance_label"] = x["stance_label"][ctx_idx1:ctx_idx2] + \
                x["stance_label"][tx_idx1:tx_idx2]
            x["stance_logit"] = th.cat(
                [x["stance_logit"][ctx_idx1:ctx_idx2], x["stance_logit"][tx_idx1:tx_idx2]], dim=0
            )
            x["discourse_label"] = x["discourse_label"][ctx_idx1:ctx_idx2] + \
                x["discourse_label"][tx_idx1:tx_idx2]
            x["discourse_logit"] = th.cat(
                [x["discourse_logit"][ctx_idx1:ctx_idx2], x["discourse_logit"][tx_idx1:tx_idx2]], dim=0
            )

        sampler = BucketSampler(
            datasets[data_type],
            group_by=["max_num_text", "num_context", "text_len", "context_len"],
            batch_size=(args.train_batch_size if data_type == "train" else args.eval_batch_size),
            shuffle=(data_type == "train"),
            drop_last=False
        )

        data_loaders[data_type] = data.DataLoader(datasets[data_type], batch_sampler=sampler, collate_fn=batchify)

    # model
    if args.mode == "flat":
        model = FlatModel(special_tokens=special_tokens, num_labels=num_labels, **vars(args))
    elif args.mode == "interval":
        model = IntervalModel(special_tokens=special_tokens, num_labels=num_labels, **vars(args))
    elif args.mode == "segmented":
        model = SegmentedModel(special_tokens=special_tokens, num_labels=num_labels, **vars(args))
    elif args.mode == "contextualized":
        model = ContextualizedModel(special_tokens=special_tokens, num_labels=num_labels, **vars(args))
    elif args.mode == "discoc":
        model = DisCOCModel(special_tokens=special_tokens, num_labels=num_labels, **vars(args))
    elif args.mode == "han":
        model = HAN(special_tokens=special_tokens, num_labels=num_labels, **vars(args))

    if args.pretrained_model_path:
        logger.info("loading pretarined model (%s)..." % (args.pretrained_model_path))
        model.load_pt(args.pretrained_model_path)
    if args.pretrained_token_type_embedding_path:
        logger.info("loading pretarined token type embeddings (%s)..." % (args.pretrained_token_type_embedding_path))
        state_dict = th.load(args.pretrained_token_type_embedding_path, map_location=th.device("cpu"))
        num_pretrained_token_type_embeddings = state_dict["weight"].size(0)

        if args.encoder == "lstm":
            token_type_embeddings = model.encoder.token_type_embeddings
        elif args.encoder == "xlnet":
            token_type_embeddings = None
        else:
            token_type_embeddings = model.encoder.model.embeddings.token_type_embeddings

        if token_type_embeddings is not None:
            r = math.ceil(token_type_embeddings.num_embeddings / num_pretrained_token_type_embeddings)
            with th.no_grad():
                token_type_embeddings.weight.data.copy_(
                    state_dict["weight"].repeat(r, 1)[:token_type_embeddings.num_embeddings]
                )

        del state_dict

    if args.encoder in ["bert", "roberta", "albert"]:
        word_embeddings = model.encoder.model.embeddings.word_embeddings
    elif args.encoder in ["xlnet"]:
        word_embeddings = model.encoder.model.word_embedding
    elif args.encoder in ["lstm"]:
        word_embeddings = model.encoder.word_embeddings
    with th.no_grad():
        nn.init.normal_(
            word_embeddings.weight.data[tokenizer.get_special_token_id(DUMMY)],
            INIT_EMB_MEAN, INIT_EMB_STD
        )
        word_embeddings.weight.data[tokenizer.get_special_token_id(DUMMY)] += \
            word_embeddings.weight[tokenizer.get_special_token_id(PAD)]
        word_embeddings.weight.data[tokenizer.get_special_token_id(CONTEXT)].copy_(
            word_embeddings.weight[tokenizer.get_special_token_id(CLS)]
        )
        word_embeddings.weight.data[tokenizer.get_special_token_id(TEXT)].copy_(
            word_embeddings.weight[tokenizer.get_special_token_id(CLS)]
        )
        if args.add_stance_token:
            for token in STANCES:
                nn.init.normal_(
                    word_embeddings.weight.data[tokenizer.get_special_token_id(token)],
                    INIT_EMB_MEAN, INIT_EMB_STD
                )
                word_embeddings.weight.data[tokenizer.get_special_token_id(token)] += \
                    word_embeddings.weight[tokenizer.get_special_token_id(SEP)]
        else:
            for token in STANCES:
                nn.init.constant_(word_embeddings.weight.data[tokenizer.get_special_token_id(token)], 0.0)

        if args.add_disco_token:
            for token in DISCOS:
                nn.init.normal_(
                    word_embeddings.weight.data[tokenizer.get_special_token_id(token)],
                    INIT_EMB_MEAN, INIT_EMB_STD
                )
                word_embeddings.weight.data[tokenizer.get_special_token_id(token)] += \
                    word_embeddings.weight[tokenizer.get_special_token_id(SEP)]
        else:
            for token in DISCOS:
                nn.init.constant_(word_embeddings.weight.data[tokenizer.get_special_token_id(token)], 0.0)

    change_dropout_rate(model, args.dropout)
    model.set_finetune(args.finetune)
    model.encoder.stance_t.requires_grad = False
    model.encoder.disco_t.requires_grad = False

    with th.no_grad():
        if not args.add_stance_embed:
            nn.init.constant_(model.encoder.stance_w1, 0.0)
            nn.init.constant_(model.encoder.stance_w2, 0.0)
            model.encoder.stance_w1.requires_grad = False
            model.encoder.stance_w2.requires_grad = False
        if not args.add_disco_embed:
            nn.init.constant_(model.encoder.disco_w1, 0.0)
            nn.init.constant_(model.encoder.disco_w2, 0.0)
            model.encoder.disco_w1.requires_grad = False
            model.encoder.disco_w2.requires_grad = False

    if args.gpu_ids and len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)
    model = model.to(device)
    logger.info(model)
    logger.info("num of total parameters: %d" % (sum(p.numel() for p in model.parameters())))
    logger.info("num of trainable parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # optimizer and losses
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    scheduler = LinearWarmupScheduler(
        num_warmup_steps=args.epochs * len(data_loaders["train"]) // 100 * 6,
        num_schedule_steps=args.epochs * len(data_loaders["train"]),
        min_percent=args.lr / 100
    )
    scheduler.set_optimizer(optimizer)

    best_losses = {dataset: INF for dataset in datasets}
    best_loss_epochs = {dataset: -1 for dataset in datasets}
    best_accs = {dataset: _INF for dataset in datasets}
    best_acc_epochs = {dataset: -1 for dataset in datasets}
    best_f1s = {dataset: _INF for dataset in datasets}
    best_f1_epochs = {dataset: -1 for dataset in datasets}
    best_accf1s = {dataset: _INF for dataset in datasets}
    best_accf1_epochs = {dataset: -1 for dataset in datasets}

    for epoch in range(args.epochs):
        for data_type, data_loader in data_loaders.items():
            if data_type == "train":
                epoch_avg_loss, results = train_epoch(
                    args, logger, writer, model, optimizer, scheduler, data_type, data_loader, device, epoch
                )
            else:
                epoch_avg_loss, results = eval_epoch(args, logger, writer, model, data_type, data_loader, device, epoch)
                save_results(results, os.path.join(args.save_model_dir, "%s_results%d.json" % (data_type, epoch)))

            logger.info("#" * 80)
            if epoch_avg_loss <= best_losses[data_type]:
                best_losses[data_type] = epoch_avg_loss
                best_loss_epochs[data_type] = epoch
                logger.info(
                    generate_log_line(
                        data_type,
                        **{
                            "best loss": "{:.4f}".format(best_losses[data_type]),
                            "best epoch": best_loss_epochs[data_type]
                        },
                    )
                )
                if args.save_best == "loss":
                    if args.gpu_ids and len(args.gpu_ids) > 1:
                        th.save(
                            model.module.state_dict(),
                            os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                            _use_new_zipfile_serialization=False
                        )
                    else:
                        th.save(
                            model.state_dict(),
                            os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                            _use_new_zipfile_serialization=False
                        )

            if results["evaluation"]["accuracy"] >= best_accs[data_type]:
                best_accs[data_type] = results["evaluation"]["accuracy"]
                best_acc_epochs[data_type] = epoch
                logger.info(
                    generate_log_line(
                        data_type,
                        **{
                            "best accuracy": "{:.4f}".format(best_accs[data_type]),
                            "best epoch": best_acc_epochs[data_type]
                        },
                    )
                )
                if args.save_best == "acc":
                    if args.gpu_ids and len(args.gpu_ids) > 1:
                        th.save(
                            model.module.state_dict(),
                            os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                            _use_new_zipfile_serialization=False
                        )
                    else:
                        th.save(
                            model.state_dict(),
                            os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                            _use_new_zipfile_serialization=False
                        )

            if results["evaluation"]["precision_recall_f1"][-1] >= best_f1s[data_type]:
                best_f1s[data_type] = results["evaluation"]["precision_recall_f1"][-1]
                best_f1_epochs[data_type] = epoch
                logger.info(
                    generate_log_line(
                        data_type,
                        **{
                            "best f1": "{:.4f}".format(best_f1s[data_type]),
                            "best epoch": best_f1_epochs[data_type]
                        },
                    )
                )
                if args.save_best == "f1":
                    if args.gpu_ids and len(args.gpu_ids) > 1:
                        th.save(
                            model.module.state_dict(),
                            os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                            _use_new_zipfile_serialization=False
                        )
                    else:
                        th.save(
                            model.state_dict(),
                            os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                            _use_new_zipfile_serialization=False
                        )

            if results["evaluation"]["accuracy"] + \
                results["evaluation"]["precision_recall_f1"][-1] >= \
                    best_accf1s[data_type]:
                best_accf1s[data_type] = \
                    results["evaluation"]["accuracy"] + \
                    results["evaluation"]["precision_recall_f1"][-1]
                best_accf1_epochs[data_type] = epoch
                logger.info(
                    generate_log_line(
                        data_type,
                        **{
                            "best accf1": "{:.4f}".format(best_accf1s[data_type]),
                            "best epoch": best_accf1_epochs[data_type]
                        },
                    )
                )
                if args.save_best == "accf1":
                    if args.gpu_ids and len(args.gpu_ids) > 1:
                        th.save(
                            model.module.state_dict(),
                            os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                            _use_new_zipfile_serialization=False
                        )
                    else:
                        th.save(
                            model.state_dict(),
                            os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                            _use_new_zipfile_serialization=False
                        )
            logger.info("#" * 80)
    logger.info("=" * 80)
    for data_type in data_loaders:
        logger.info(
            generate_log_line(
                data_type,
                **{
                    "best loss": "{:.4f}".format(best_losses[data_type]),
                    "best epoch": best_loss_epochs[data_type]
                },
            )
        )
        logger.info(
            generate_log_line(
                data_type,
                **{
                    "best accuracy": "{:.4f}".format(best_accs[data_type]),
                    "best epoch": best_acc_epochs[data_type]
                },
            )
        )
        logger.info(
            generate_log_line(
                data_type,
                **{
                    "best f1": "{:.4f}".format(best_f1s[data_type]),
                    "best epoch": best_f1_epochs[data_type]
                },
            )
        )
        logger.info(
            generate_log_line(
                data_type,
                **{
                    "best accf1": "{:.4f}".format(best_accf1s[data_type]),
                    "best epoch": best_accf1_epochs[data_type]
                },
            )
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="numer of processors"
    )

    # data config
    parser.add_argument(
        "--label_map",
        type=str,
        default="argument_impact_label_map_3",
        choices=[
            "argument_impact_label_map_3",
        ],
        help="label map"
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="../data/arg_impact/train.jsonl",
        help="training DiscoChainDataset path"
    )
    parser.add_argument(
        "--valid_dataset_path",
        type=str,
        default="../data/arg_impact/valid.jsonl",
        help="validation DiscoChainDataset path"
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="../data/arg_impact/test.jsonl",
        help="test DiscoChainDataset path"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="",
        help="model path of the pretrained model"
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="../dumps/argimpact",
        help="model dir to save models"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=100,
        help="the maximum length of text and context"
    )
    parser.add_argument(
        "--max_num_text",
        type=int,
        default=1,
        help="the maximum number of text"
    )
    parser.add_argument(
        "--max_num_context",
        type=int,
        default=1,
        help="the maximum number of contexts"
    )
    parser.add_argument(
        "--cls_last",
        action="store_true",
        help="whether to exchange the [CLS] and [SEP] for bert/roberta/albert"
    )
    parser.add_argument(
        "--add_stance_token",
        action="store_true",
        help="add stance between two claims"
    )
    parser.add_argument(
        "--add_disco_token",
        action="store_true",
        help="add discourse labels between two claims"
    )
    parser.add_argument(
        "--add_stance_embed",
        action="store_true",
        help="add stance between two claims"
    )
    parser.add_argument(
        "--add_disco_embed",
        action="store_true",
        help="add discourse labels between two claims"
    )

    # training config
    parser.add_argument(
        "--gpu_ids",
        type=str2list,
        default=None,
        help="gpu ids"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="epochs of training"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="batch size of training"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=10,
        help="batch size of evaluation"
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=100,
        help="printing log every K batchs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="learning rate for the optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="weight decay"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=2.0,
        help="max grad norm for gradient clipping"
    )
    parser.add_argument(
        "--save_best",
        type=str,
        default="f1",
        choices=["loss", "acc", "f1", "accf1"],
        help="the criteria to save best models"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce", choices=["ce"],
        help="loss function"
    )

    # model config
    parser.add_argument(
        "--mode",
        type=str,
        choices=["han", "flat", "interval", "segmented", "contextualized", "discoc"],
        help="han, flat, interval, segmented, contextualized, or discoc encoding"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="lstm",
        choices=["lstm", "bert", "roberta", "albert", "xlnet"],
        help="the encoder"
    )
    parser.add_argument(
        "--add_matching",
        action="store_true",
        help="whether add bilateral multi-perspective matching"
    )
    parser.add_argument(
        "--add_fusion",
        action="store_true",
        help="whether add gated fusion"
    )
    parser.add_argument(
        "--add_conv",
        action="store_true",
        help="whether add a convolutional layer"
    )
    parser.add_argument(
        "--add_trans",
        action="store_true",
        help="whether add a transformer layer"
    )
    parser.add_argument(
        "--add_gru",
        action="store_true",
        help="whether add a gru layer"
    )
    parser.add_argument(
        "--pretrained_token_type_embedding_path",
        type=str,
        default="",
        help="pretrained token type embeddings path"
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default="full",
        choices=["full", "layers", "last", "type", "none"],
        help="how to finetune the encoder"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="hidden dimension"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="number of lstm layers"
    )
    parser.add_argument(
        "--num_perspectives",
        type=int,
        default=16,
        help="number of perspectives for bimpm"
    )
    parser.add_argument(
        "--conv_filters",
        type=int,
        default=64,
        help="number of filters for convolutional layers"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="leaky_relu",
        choices=["relu", "tanh", "softmax", "sigmoid", "leaky_relu", "prelu", "gelu"],
        help="activation function type"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout for neural networks"
    )
    parser.add_argument(
        "--word2vec_file",
        type=str,
        default="../data/glove/glove_kialo.txt",
        help="word2vec file for lstm"
    )
    parser.add_argument(
        "--bert_dir",
        type=str,
        default="../data/pretrained_lm/bert",
        help="bert dir location, including vocab and model files"
    )
    parser.add_argument(
        "--roberta_dir",
        type=str,
        default="../data/pretrained_lm/roberta",
        help="roberta dir location, including vocab and model files"
    )
    parser.add_argument(
        "--xlnet_dir",
        type=str,
        default="../data/pretrained_lm/xlnet",
        help="xlnet dir location, including vocab and model files"
    )
    parser.add_argument(
        "--albert_dir",
        type=str,
        default="../data/pretrained_lm/albert",
        help="albert dir location, including vocab and model files"
    )

    args = parser.parse_args()

    ts = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args.save_model_dir = os.path.join(
        args.save_model_dir, "%s_%s_text(%d)_context(%d)_relation(%d)_seed(%d)_%s" % (
            ("STE_" if args.add_stance_embed else "") + ("DSE_" if args.add_disco_embed else "") +
            ("STT_" if args.add_stance_token else "") + ("DST_" if args.add_disco_token else "") +
            ("BM_" if args.add_matching else "") + ("GF_" if args.add_fusion else "") +
            ("CONV_" if args.add_conv else "") +
            ("TRANS_" if args.add_trans else "") +
            ("GRU_" if args.add_gru else "") +
            (str.capitalize(args.mode)),
            args.encoder,
            args.max_num_text,
            args.max_num_context,
            int(args.label_map.rsplit("_", 1)[1]),
            args.seed,
            ts
        )
    )
    os.makedirs(args.save_model_dir, exist_ok=True)

    # save config
    save_config(args, os.path.join(args.save_model_dir, "config.json"))

    # build logger
    logger = init_logger(
        log_file=os.path.join(args.save_model_dir, "log.txt"),
        log_tag=("STE_" if args.add_stance_embed else "") + ("DSE_" if args.add_disco_embed else "") +
            ("STT_" if args.add_stance_token else "") + ("DST_" if args.add_disco_token else "") +
            ("BM_" if args.add_matching else "") + ("GF_" if args.add_fusion else "") +
            ("CONV_" if args.add_conv else "") +
            ("TRANS_" if args.add_trans else "") +
            ("GRU_" if args.add_gru else "") +
            (str.capitalize(args.mode))
    )

    # build writer
    writer = SummaryWriter(args.save_model_dir)

    # train
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args, logger, writer)

    close_logger(logger)
