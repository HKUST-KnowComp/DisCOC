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
from dataset import *
from predictor import *

INF = 1e30
_INF = -1e30
BUCKET_DIFF = 2


def eval_epoch(args, logger, writer, model, data_type, data_loader, device, epoch):
    model.eval()
    epoch_step = len(data_loader)
    total_step = args.epochs * epoch_step
    total_cnt = 0
    total_ce = 0.0

    results = {
        "data": {
            "id": list(),
            "label": list(),
            "labels": list()
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
        }
    }

    with th.no_grad():
        for batch_id, batch in enumerate(data_loader):
            step = epoch * epoch_step + batch_id
            _id, disco, mask, labels, label = batch
            bsz = len(_id)

            results["data"]["id"].extend(_id)
            results["data"]["labels"].extend(labels.numpy())
            results["data"]["label"].extend(label.numpy())

            disco = disco.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            label = label.to(device)

            output = model(disco, mask)
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

            if args.loss == "ce":
                avg_loss = avg_ce
            else:
                raise NotImplementedError("Error: loss=%s is not supported now." % (args.loss))

            if writer:
                writer.add_scalar("%s/loss" % (data_type), avg_loss, step)
                writer.add_scalar("%s/ce" % (data_type), avg_ce, step)

            if False and logger and (batch_id == epoch_step - 1 or batch_id % args.print_every == 0):
                logger.info(
                    generate_log_line(
                        data_type,
                        epoch=epoch,
                        total_epochs=args.epochs,
                        step=batch_id,
                        total_steps=epoch_step,
                        **{
                            "\n" + " " * 26 + "loss":
                                avg_loss,
                            "ce":
                                avg_ce,
                            "\n" + " " * 26 + "gold":
                                ", ".join(["{:4.4f}".format(x) for x in results["data"]["labels"][-1]]),
                            "\n" + " " * 26 + "pred":
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

        if writer:
            writer.add_scalar("%s/loss-epoch" % (data_type), epoch_avg_loss, epoch)
            writer.add_scalar("%s/ce-epoch" % (data_type), epoch_avg_ce, epoch)

        if logger:
            logger.info("-" * 80)
            logger.info(
                generate_log_line(
                    data_type + "-best",
                    epoch=epoch,
                    total_epochs=args.epochs,
                    **{
                        "\n" + " " * 26 + "loss-epoch":
                            avg_loss,
                        "ce-epoch":
                            epoch_avg_loss,
                        "\n" + " " * 26 + "accuracy":
                            pprint.pformat(results["evaluation"]["accuracy"]).replace("\n", "\n" + " " * 26),
                        "\n" + " " * 26 + "precision_recall_f1":
                            pprint.pformat(results["evaluation"]["precision_recall_f1"]).replace("\n", "\n" + " " * 26)
                    },
                )
            )

    gc.collect()
    return epoch_avg_loss, results


def train_epoch(args, logger, writer, model, optimizer, scheduler, data_type, data_loader, device, epoch):
    model.train()
    epoch_step = len(data_loader)
    total_step = args.epochs * epoch_step
    total_cnt = 0
    total_ce = 0.0

    results = {
        "data": {
            "id": list(),
            "label": list(),
            "labels": list()
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
        }
    }

    for batch_id, batch in enumerate(data_loader):
        step = epoch * epoch_step + batch_id
        _id, disco, mask, labels, label = batch
        bsz = len(_id)

        results["data"]["id"].extend(_id)
        results["data"]["labels"].extend(labels.numpy())
        results["data"]["label"].extend(label.numpy())

        disco = disco.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        label = label.to(device)

        output = model(disco, mask)
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

        if logger and (batch_id == epoch_step - 1 or batch_id % args.print_every == 0):
            logger.info(
                generate_log_line(
                    data_type,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    step=batch_id,
                    total_steps=epoch_step,
                    **{
                        "\n" + " " * 26 + "loss":
                            avg_loss,
                        "ce":
                            avg_ce,
                        "\n" + " " * 26 + "gold":
                            ", ".join(["{:4.4f}".format(x) for x in results["data"]["labels"][-1]]),
                        "\n" + " " * 26 + "pred":
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

    if writer:
        writer.add_scalar("%s/loss-epoch" % (data_type), epoch_avg_loss, epoch)
        writer.add_scalar("%s/ce-epoch" % (data_type), epoch_avg_ce, epoch)

    if logger:
        logger.info("-" * 80)
        logger.info(
            generate_log_line(
                data_type + "-best",
                epoch=epoch,
                total_epochs=args.epochs,
                **{
                    "\n" + " " * 26 + "loss-epoch":
                        avg_loss,
                    "ce-epoch":
                        epoch_avg_loss,
                    "\n" + " " * 26 + "accuracy":
                        pprint.pformat(results["evaluation"]["accuracy"]).replace("\n", "\n" + " " * 26),
                    "\n" + " " * 26 + "precision_recall_f1":
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

    # load data
    datasets = OrderedDict(
        {
            "train": DiscoDataset().load_jsonl(args.train_dataset_path),
            "valid": DiscoDataset().load_jsonl(args.valid_dataset_path),
            "test": DiscoDataset().load_jsonl(args.test_dataset_path)
        }
    )

    logger.info("train:valid:test = %d:%d:%d" % (len(datasets["train"]), len(datasets["valid"]), len(datasets["test"])))

    # build data loaders
    data_loaders = OrderedDict()
    batchify = partial(
        DiscoDataset.batchify,
        label_map=label_map,
        pre_pad=True
    )
    for data_type in datasets:
        for x in datasets[data_type]:
            x["discourse"] = x["discourse"][-args.max_num_pairs:]
            x["discourse_len"] = sum([math.ceil(c.size(0) / BUCKET_DIFF) * BUCKET_DIFF for c in x["discourse"]])

        sampler = BucketSampler(
            datasets[data_type],
            group_by=["discourse_len"],
            batch_size=(args.train_batch_size if data_type == "train" else args.eval_batch_size),
            shuffle=(data_type == "train"),
            drop_last=False
        )

        data_loaders[data_type] = data.DataLoader(datasets[data_type], batch_sampler=sampler, collate_fn=batchify)

    # model
    if args.predictor == "mlp":
        model = MLPPredictor(
            input_dim=datasets["train"].data[0]["discourse"].size(-1),
            num_labels=num_labels, num_segments=args.max_num_pairs, **vars(args)
        )
    elif args.predictor == "lstm":
        model = LSTMPredictor(
            input_dim=datasets["train"].data[0]["discourse"].size(-1),
            num_labels=num_labels, num_segments=args.max_num_pairs, **vars(args)
        )
    elif args.predictor == "cnn":
        model = CNNPredictor(
            input_dim=datasets["train"].data[0]["discourse"].size(-1),
            num_labels=num_labels, num_segments=args.max_num_pairs, **vars(args)
        )
    elif args.predictor == "transformer":
        model = TransformerPredictor(
            input_dim=datasets["train"].data[0]["discourse"].size(-1),
            num_labels=num_labels, num_segments=args.max_num_pairs, **vars(args)
        )

    if args.gpu_ids and len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)
    model = model.to(device)
    logger.info(model)
    logger.info("num of total parameters: %d" % (sum(p.numel() for p in model.parameters())))
    logger.info("num of trainable parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # optimizer and losses
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=args.epochs * len(data_loaders["train"]) // 100 * 6,
    #     num_schedule_steps=args.epochs * len(data_loaders["train"]),
    #     min_percent=args.lr / 100
    # )
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
            "argument_impact_label_map_3"
        ],
        help="label map"
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="../data/arg_impact/train_discourse.jsonl",
        help="training discorse output path"
    )
    parser.add_argument(
        "--valid_dataset_path",
        type=str,
        default="../data/arg_impact/valid_discourse.jsonl",
        help="validation discorse output path"
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="../data/arg_impact/test_discourse.jsonl",
        help="test discorse output path"
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="../dumps/argimpact",
        help="model dir to save models"
    )
    parser.add_argument(
        "--max_num_pairs",
        type=int,
        default=1,
        help="the maximum number of pairs"
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
        default=100,
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
        "--predictor",
        type=str,
        default="lstm", choices=["mlp", "lstm", "cnn", "transformer"],
        help="the predictor"
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

    args = parser.parse_args()

    ts = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args.save_model_dir = os.path.join(
        args.save_model_dir, "%s_pairs(%d)_relation(%d)_seed(%d)_%s" %
        (args.predictor, args.max_num_pairs, int(args.label_map.rsplit("_", 1)[1]), args.seed, ts)
    )
    os.makedirs(args.save_model_dir, exist_ok=True)

    # save config
    save_config(args, os.path.join(args.save_model_dir, "config.json"))

    # build logger
    logger = init_logger(log_file=os.path.join(args.save_model_dir, "log.txt"), log_tag=(args.predictor))

    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    # console = logging.StreamHandler()
    # console.setFormatter(fmt)
    # logger.addHandler(console)
    # logfile = logging.FileHandler(
    #     os.path.join(args.save_model_dir, "log.txt"), 'w'
    # )
    # logfile.setFormatter(fmt)
    # logger.addHandler(logfile)

    # build writer
    writer = SummaryWriter(args.save_model_dir)

    # train
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args, logger, writer)

    close_logger(logger)
