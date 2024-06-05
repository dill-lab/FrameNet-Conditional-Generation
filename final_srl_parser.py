from __future__ import division
import pickle
import sys

'''
Reads XML files containing FrameNet 1.$VERSION annotations, and converts them to a CoNLL 2009-like format.
'''
import codecs
import os

import importlib
importlib.reload(sys)

from tqdm import tqdm
import random
import time
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as et
from collections import Counter

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import re
from scipy import stats
from itertools import combinations
import torch
from torch import nn
from transformers import BertForTokenClassification, BertTokenizer, \
                         TrainingArguments, DataCollatorForTokenClassification, Trainer,\
                         EarlyStoppingCallback, AutoTokenizer, AutoModelForTokenClassification
import evaluate
from datasets import load_metric, Dataset
import argparse

torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)
# world_size = torch.cuda.device_count()

model_checkpoint = 'SpanBERT/spanbert-large-cased'
tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_len = 1024
batch_size = 8
results_f1 = []

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class SentenceFEs(object):

    def __init__(self, lu_id, sent_id, text, core_fes):
        self.lu_id = lu_id
        self.id = sent_id
        self.text = text
        self.fes = []
        self.target_stinds = []
        self.target_eninds = []
        self.core_fes = core_fes

    def add_target(self, start, end):
        self.target_stinds.append(start)
        self.target_eninds.append(end)

    def add_fe(self, label, label_id, start, end):
        fe = {}
        fe['id'] = label_id
        fe['name'] = label
        fe['startind'] = start
        fe['endind'] = end
        fe['is_core'] = (label in self.core_fes)
        self.fes.append(fe)

    def add_fe_phrasetype(self, start, phrase_type):
        for fe in self.fes:
            if fe['startind'] == start:
                fe['phrase_type'] = phrase_type
                break

    def sort_target_inds(self):
        self.target_stinds = sorted(self.target_stinds)
        self.target_eninds = sorted(self.target_eninds)

# load pickle files
with open("frame2lus.pickle", "rb") as f:
    frame2lus = pickle.load(f)
with open("lu2frame.pickle", "rb") as f:
    lu2frame = pickle.load(f)
with open("lu2sents.pickle", "rb") as f:
    lu2sents = pickle.load(f)
with open("id2lu.pickle", "rb") as f:
    id2lu = pickle.load(f)
with open("id2frame.pickle", "rb") as f:
    id2frame = pickle.load(f)
with open("id2fe.pickle", "rb") as f:
    id2fe = pickle.load(f)
with open("candidate_fes.pickle", "rb") as f:
    candidate_fes = pickle.load(f)
with open("frame2fes.pickle", "rb") as f:
    frame2fes = pickle.load(f)
with open("fe2frame.pickle", "rb") as f:
    fe2frame = pickle.load(f)
with open("fe_names.pickle", "rb") as f:
    fe_names = pickle.load(f)


# FEs_B = ["B-"+fe_id for fe_id in id2fe]
# FEs_I = ["I-"+fe_id for fe_id in id2fe]
FEs_B = ["B-"+fe_name for fe_name in fe_names]
FEs_I = ["I-"+fe_name for fe_name in fe_names]
# FEs_B = ["B-"+fe_name for fe_name in fe_names_v]
# FEs_I = ["I-"+fe_name for fe_name in fe_names_v]
labels = ["O"] + FEs_B + FEs_I
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
CLS = "[CLS]"
SEP = "[SEP]"
LU_START = "[unused1]"
LU_END = "[unused2]"
frame2labelids = {}
for frame_id in tqdm(id2frame.keys()):
    frame_fe_names = [id2fe[fe_id] for fe_id in frame2fes[frame_id]]
    frame_labels = ['O'] + ['B-'+fe_name for fe_name in frame_fe_names] + ['I-'+fe_name for fe_name in frame_fe_names]
    frame2labelids[frame_id] = [label2id[label] for label in frame_labels]


class TrainSentence:
    def __init__(self, sent_id, frame_id, text, fe_start_inds, fe_end_inds, lu_start_ind, lu_end_ind, lu_id, fe_ids):
        self.frame_id = frame_id
        self.text = text
        self.fe_start_inds = fe_start_inds
        self.fe_end_inds = fe_end_inds
        self.lu_start_ind = lu_start_ind
        self.lu_end_ind = lu_end_ind
        self.lu_id = lu_id
        self.fe_ids = fe_ids
        self.skip = False

    def preprocess(self):
        lu_id = self.lu_id
        frame_id = self.frame_id
        text = self.text
        fe_start_inds = self.fe_start_inds
        fe_end_inds = self.fe_end_inds
        lu_start_ind = self.lu_start_ind
        lu_end_ind = self.lu_end_ind
        fe_ids = self.fe_ids

        # truncate sentence
        text = text[:max_len]

        tokens = [CLS]
        tokenized_ind = 0
        # ignore special tokens in labels
        label_ids = [-100]
        for ind in range(len(text)):
            if ind in fe_start_inds:
                # span before FE
                span_tokens = tokenizer.tokenize(text[tokenized_ind:ind])
                tokens += span_tokens
                # only label first subword of a given word
                label_ids += [-100 if token[:2] == "##" else label2id['O'] for token in span_tokens]
                tokenized_ind = ind
            elif ind in fe_end_inds:
                # FE span
                fe_id = fe_ids[fe_end_inds.index(ind)]
                span_tokens = tokenizer.tokenize(text[tokenized_ind:(ind+1)])
                if len(span_tokens) > 0:
                    tokens += span_tokens
                    # first (sub)word in FE span
                    # label_ids.append(label2id['B-'+fe_id])
                    label_ids.append(label2id['B-'+id2fe[fe_id]])
                    # remaining words in FE span
                    # rem_words_label = label2id['I-'+fe_id]
                    rem_words_label = label2id['I-'+id2fe[fe_id]]
                    # only label first subword of a given word
                    label_ids += [-100 if token[:2] == "##" else rem_words_label for token in span_tokens[1:]]
                tokenized_ind = ind+1
            elif ind == lu_start_ind:
                # span before LU
                span_tokens = tokenizer.tokenize(text[tokenized_ind:ind])
                tokens += span_tokens
                label_ids += [-100 if token[:2] == "##" else label2id['O'] for token in span_tokens]
                tokens.append(LU_START)
                # ignore special tokens
                label_ids.append(-100)
                tokenized_ind = lu_start_ind
            elif ind == lu_end_ind:
                # LU span
                span_tokens = tokenizer.tokenize(text[tokenized_ind:(ind+1)])
                tokens += span_tokens
                label_ids += [-100 if token[:2] == "##" else label2id['O'] for token in span_tokens]
                tokens.append(LU_END)
                label_ids.append(-100)
                tokenized_ind = lu_end_ind+1
        # final span
        span_tokens = tokenizer.tokenize(text[tokenized_ind:])
        tokens += span_tokens
        label_ids += [-100 if token[:2] == "##" else label2id['O'] for token in span_tokens]
        tokens.append(SEP)
        label_ids.append(-100)

        lu_tokens = tokenizer.tokenize(id2lu[lu_id])
        tokens += lu_tokens
        tokens.append(SEP)
        # ignore appended LU name
        label_ids += [-100] * (len(lu_tokens)+1)

        frame_tokens = tokenizer.tokenize(id2frame[frame_id])
        tokens += frame_tokens
        tokens.append(SEP)
        # ignore appended frame name
        label_ids += [-100] * (len(frame_tokens)+1)

        possible_fes = [id2fe[fe_id] for fe_id in frame2fes[frame_id]]
        possible_fe_tokens = []
        for fe in possible_fes:
            possible_fe_tokens += tokenizer.tokenize(fe)
            possible_fe_tokens.append('[unused3]')
        # do not insert unused token at the end of last FE
        tokens += possible_fe_tokens[:-1]
        tokens.append(SEP)
        # ignore appended possible FE names
        label_ids += [-100] * (len(possible_fe_tokens))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        self.tokens = tokens
        self.input_ids = input_ids
        self.labels = label_ids
        self.kept_labels = frame2labelids[frame_id] + [0] * (len(input_ids)-len(frame2labelids[frame_id]))
        self.frame_id = int(frame_id)

def create_data_one_sent(sentanno):
    lu_id = sentanno.lu_id
    text = sentanno.text 
    # no LU in sentence
    if len(sentanno.target_stinds) == 0:
        return None 
    lu_start_ind = sentanno.target_stinds[0]
    lu_end_ind = sentanno.target_eninds[0]
    frame_id = lu2frame[lu_id]
    fe_start_inds, fe_end_inds, fe_ids = [], [], []
    for fe in sentanno.fes:
        fe_start_inds.append(fe['startind'])
        fe_end_inds.append(fe['endind'])
        fe_ids.append(fe['id'])
    sent = TrainSentence(sentanno.id, frame_id, text, fe_start_inds, fe_end_inds, lu_start_ind, lu_end_ind, lu_id, fe_ids)
    sent.preprocess()
    return sent

def create_data_one_sent_empty(lu_id, sentanno, text_out, fe_inds, lu_inds):
    if len(fe_inds) == 0:
        return []
    text = text_out
    lu_start, lu_end = lu_inds
    fes = sorted(sentanno.fes, key=lambda d: d['startind'])
    sents = []
    for fe in fes:
        fe_start, fe_end = fe_inds[counter]
        frame = id2frame[lu2frame[lu_id]]
        fe_id = fe['id']
        sent = TrainSentence(frame, text, fe_start, fe_end, lu_start, lu_end, lu_id, fe_id)
        sent.preprocess()
        sents.append(sent)
    return sents

def create_data(data_sentanno):
    data = []
    count = 0
    # only print tqdm messages once if using multi-GPU
    if args.local_rank == 0:
        for sentanno in tqdm(data_sentanno):
            sent = create_data_one_sent(sentanno)
            if sent is not None:
                data.append(sent)
                count += 1
    else:
        for sentanno in data_sentanno:
            sent = create_data_one_sent(sentanno)
            if sent is not None:
                data.append(sent)
                count += 1
    return data


def create_inputs_targets(data):
    dataset_dict = {
        "input_ids": [],
        "labels": [],
        "frame_id": [],
    }
    for item in data:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    # dataset_dict["labels"] = dataset_dict.pop("label_id")
    return dataset_dict

def calc_f(scores):
    tp, fp, fn = scores
    pr =  tp / (tp + fp + 1e-13)
    re = tp / (tp + fn + 1e-13)
    f1 = 2.0 * (pr * re) / (pr + re + 1e-13)
    return pr, re, f1

def token_level_eval(labels, preds):
    # token-level F1
    notanfeid = 'O'
    wtp = wfp = wfn = 0.0
    for i in range(len(labels)):
        if labels[i] == preds[i]:
            if labels[i] != notanfeid:
                wtp += 1
        elif labels[i] == notanfeid and preds[i] != notanfeid:
            wfp += 1
        elif preds[i] == notanfeid and labels[i] != notanfeid:
            wfn += 1
        else:
            wfp += 1
            wfn += 1

    return wtp, wfp, wfn

def get_fe2inds(bio_tags):
    fe2inds = {}
    span_start = -1
    span_end = -1
    fe_id = None
    for i in range(len(bio_tags)):
        tag = bio_tags[i]
        if tag == 'O':
            continue
        # label is not "O"
        # start of new FE span
        # merge adjacent spans with the same label
        if i == 0 or (tag == 'B' and tag[2:] != fe_id) or bio_tags[i-1] == 'O':
            fe_id = tag[2:]
            span_start = i  
        # end of current span
        # merge adjacent spans with the same label
        if i == len(bio_tags)-1 or bio_tags[i+1][0] == 'O' or (bio_tags[i+1][0] == 'B' and bio_tags[i+1][2:] != fe_id):
            span_end = i+1
            if fe_id not in fe2inds:
                fe2inds[fe_id] = []
            fe2inds[fe_id].append((span_start,span_end))
    return fe2inds

def labeled_eval(labels, preds):
    # fe2inds has the form 
    # {'fe_id'/'fe_name':[(start,end),(start,end)], 'fe_id'/'fe_name':[(start,end),(start,end),...],...}
    # since one sentence can have multiple spans of the same FE type
    labels_fe2inds = get_fe2inds(labels)
    preds_fe2inds = get_fe2inds(preds)

    match = predicted = gold = 0.0
    ltp = lfp = lfn = 0.0
    # labeled spans
    for goldfe, inds in labels_fe2inds.items():
        for startend in inds:
            if goldfe in preds_fe2inds and startend in preds_fe2inds[goldfe]:
                # ltp += 1.0 if goldfe in core_fe_names else 0.5
                ltp += 1.0
            else:
                # lfn += 1.0 if goldfe in core_fe_names else 0.5
                lfn += 1.0
    for predfe, inds in preds_fe2inds.items():
        for startend in inds:
            if predfe not in labels_fe2inds or startend not in labels_fe2inds[predfe]:
                # lfp += 1.0 if predfe in core_fe_names else 0.5
                lfp += 1.0

    return ltp, lfp, lfn

def labeled_eval_weighted(sentanno, labels, preds):
    # fe2inds has the form 
    # {'fe_id'/'fe_name':[(start,end),(start,end)], 'fe_id'/'fe_name':[(start,end),(start,end),...],...}
    # since one sentence can have multiple spans of the same FE type
    labels_fe2inds = get_fe2inds(labels)
    preds_fe2inds = get_fe2inds(preds)
    core_fe_names = set(sentanno.core_fes)

    match = predicted = gold = 0.0
    ltp = lfp = lfn = 0.0
    # labeled spans
    for goldfe, inds in labels_fe2inds.items():
        for startend in inds:
            if goldfe in preds_fe2inds and startend in preds_fe2inds[goldfe]:
                ltp += 1.0 if goldfe in core_fe_names else 0.5
                # ltp += 1.0
            else:
                lfn += 1.0 if goldfe in core_fe_names else 0.5
                # lfn += 1.0
    for predfe, inds in preds_fe2inds.items():
        for startend in inds:
            if predfe not in labels_fe2inds or startend not in labels_fe2inds[predfe]:
                lfp += 1.0 if predfe in core_fe_names else 0.5
                # lfp += 1.0

    return ltp, lfp, lfn

def labeled_eval_core(sentanno, labels, preds):
    # fe2inds has the form 
    # {'fe_id'/'fe_name':[(start,end),(start,end)], 'fe_id'/'fe_name':[(start,end),(start,end),...],...}
    # since one sentence can have multiple spans of the same FE type
    labels_fe2inds = get_fe2inds(labels)
    preds_fe2inds = get_fe2inds(preds)
    core_fe_names = set(sentanno.core_fes)

    match = predicted = gold = 0.0
    ltp = lfp = lfn = 0.0
    # labeled spans
    for goldfe, inds in labels_fe2inds.items():
        for startend in inds:
            if goldfe in preds_fe2inds and startend in preds_fe2inds[goldfe]:
                ltp += 1.0 if goldfe in core_fe_names else 0
                # ltp += 1.0
            else:
                lfn += 1.0 if goldfe in core_fe_names else 0
                # lfn += 1.0
    for predfe, inds in preds_fe2inds.items():
        for startend in inds:
            if predfe not in labels_fe2inds or startend not in labels_fe2inds[predfe]:
                lfp += 1.0 if predfe in core_fe_names else 0
                # lfp += 1.0

    return ltp, lfp, lfn

def labeled_eval_noncore(sentanno, labels, preds):
    # fe2inds has the form 
    # {'fe_id'/'fe_name':[(start,end),(start,end)], 'fe_id'/'fe_name':[(start,end),(start,end),...],...}
    # since one sentence can have multiple spans of the same FE type
    labels_fe2inds = get_fe2inds(labels)
    preds_fe2inds = get_fe2inds(preds)
    core_fe_names = set(sentanno.core_fes)

    match = predicted = gold = 0.0
    ltp = lfp = lfn = 0.0
    # labeled spans
    for goldfe, inds in labels_fe2inds.items():
        for startend in inds:
            if goldfe in preds_fe2inds and startend in preds_fe2inds[goldfe]:
                ltp += 1.0 if goldfe not in core_fe_names else 0
                # ltp += 1.0
            else:
                lfn += 1.0 if goldfe not in core_fe_names else 0
                # lfn += 1.0
    for predfe, inds in preds_fe2inds.items():
        for startend in inds:
            if predfe not in labels_fe2inds or startend not in labels_fe2inds[predfe]:
                lfp += 1.0 if predfe not in core_fe_names else 0
                # lfp += 1.0

    return ltp, lfp, lfn


def compute_metrics(pred):
    logits, labels = pred
    with open('SRL_data_val_sentanno.pickle', 'rb') as f:
        data_val = pickle.load(f)
    frame_ids = [lu2frame[sentanno.lu_id] for sentanno in data_val]
    kept_labels = [frame2labelids[frame_id] for frame_id in frame_ids]
    logits = torch.from_numpy(logits)
    mask = torch.zeros_like(logits)
    for i in range(logits.shape[0]):
        mask[i, :, kept_labels[i]] = 1
    # set logits of masked labels to ignore_index
    masked_logits = torch.where(mask.bool(), logits, torch.tensor(-100)).numpy()

    predictions = masked_logits.argmax(axis=-1)
    # predictions, labels = pred

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # scores has shape [(tp,fp,fn), (tp,fp,fn), ...]
    scores = [labeled_eval(true_labels[i],true_predictions[i]) for i in range(len(labels))]
    scores = [sum([score[i] for score in scores]) for i in range(3)]
    precision, recall, f1 = calc_f(scores)

    results_f1.append(f1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def compute_metrics_weighted(pred):
    logits, labels = pred
    with open('SRL_data_val_sentanno.pickle', 'rb') as f:
        data_val = pickle.load(f)
    frame_ids = [lu2frame[sentanno.lu_id] for sentanno in data_val]
    kept_labels = [frame2labelids[frame_id] for frame_id in frame_ids]
    logits = torch.from_numpy(logits)
    mask = torch.zeros_like(logits)
    for i in range(logits.shape[0]):
        mask[i, :, kept_labels[i]] = 1
    # set logits of masked labels to ignore_index
    masked_logits = torch.where(mask.bool(), logits, torch.tensor(-100)).numpy()

    predictions = masked_logits.argmax(axis=-1)
    # predictions, labels = pred

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    scores = [labeled_eval_weighted(data_val[i],true_labels[i],true_predictions[i]) for i in range(len(data_val))]
    scores = [sum([score[i] for score in scores]) for i in range(3)]
    precision, recall, f1 = calc_f(scores)

    results_f1.append(f1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def compute_metrics_test(pred):
    logits, labels = pred
    with open('SRL_data_test_sentanno.pickle', 'rb') as f:
        data_test = pickle.load(f)
    frame_ids = [lu2frame[sentanno.lu_id] for sentanno in data_test]
    kept_labels = [frame2labelids[frame_id] for frame_id in frame_ids]
    logits = torch.from_numpy(logits)
    mask = torch.zeros_like(logits)
    for i in range(logits.shape[0]):
        mask[i, :, kept_labels[i]] = 1
    # set logits of masked labels to ignore_index
    masked_logits = torch.where(mask.bool(), logits, torch.tensor(-100)).numpy()

    predictions = masked_logits.argmax(axis=-1)
    # predictions, labels = pred
    for i, pred_sent in enumerate(predictions):
        frame_id = lu2frame[data_test[i].lu_id]
        for pred in pred_sent:
            if pred.item() != -100 and pred.item() not in set(frame2labelids[frame_id]):
                print('pred invalid FE')


    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    scores = [labeled_eval(true_labels[i],true_predictions[i]) for i in range(len(labels))]
    scores = [sum([score[i] for score in scores]) for i in range(3)]
    precision, recall, f1 = calc_f(scores)

    results_f1.append(f1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def compute_metrics_test_weighted(pred):
    logits, labels = pred
    with open('SRL_data_test_sentanno.pickle', 'rb') as f:
        data_test = pickle.load(f)
    frame_ids = [lu2frame[sentanno.lu_id] for sentanno in data_test]
    kept_labels = [frame2labelids[frame_id] for frame_id in frame_ids]
    logits = torch.from_numpy(logits)
    mask = torch.zeros_like(logits)
    for i in range(logits.shape[0]):
        mask[i, :, kept_labels[i]] = 1
    # set logits of masked labels to ignore_index
    masked_logits = torch.where(mask.bool(), logits, torch.tensor(-100)).numpy()

    predictions = masked_logits.argmax(axis=-1)
    
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    scores = [labeled_eval_weighted(data_test[i],true_labels[i],true_predictions[i]) for i in range(len(data_test))]
    scores = [sum([score[i] for score in scores]) for i in range(3)]
    precision, recall, f1 = calc_f(scores)

    results_f1.append(f1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def compute_metrics_test_core(pred):
    logits, labels = pred
    with open('SRL_data_test_sentanno.pickle', 'rb') as f:
        data_test = pickle.load(f)
    frame_ids = [lu2frame[sentanno.lu_id] for sentanno in data_test]
    kept_labels = [frame2labelids[frame_id] for frame_id in frame_ids]
    logits = torch.from_numpy(logits)
    mask = torch.zeros_like(logits)
    for i in range(logits.shape[0]):
        mask[i, :, kept_labels[i]] = 1
    # set logits of masked labels to ignore_index
    masked_logits = torch.where(mask.bool(), logits, torch.tensor(-100)).numpy()

    predictions = masked_logits.argmax(axis=-1)
    
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    scores = [labeled_eval_core(data_test[i],true_labels[i],true_predictions[i]) for i in range(len(data_test))]
    scores = [sum([score[i] for score in scores]) for i in range(3)]
    precision, recall, f1 = calc_f(scores)

    results_f1.append(f1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def compute_metrics_test_noncore(pred):
    logits, labels = pred
    with open('SRL_data_test_sentanno.pickle', 'rb') as f:
        data_test = pickle.load(f)
    frame_ids = [lu2frame[sentanno.lu_id] for sentanno in data_test]
    kept_labels = [frame2labelids[frame_id] for frame_id in frame_ids]
    logits = torch.from_numpy(logits)
    mask = torch.zeros_like(logits)
    for i in range(logits.shape[0]):
        mask[i, :, kept_labels[i]] = 1
    # set logits of masked labels to ignore_index
    masked_logits = torch.where(mask.bool(), logits, torch.tensor(-100)).numpy()

    predictions = masked_logits.argmax(axis=-1)
    
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    scores = [labeled_eval_noncore(data_test[i],true_labels[i],true_predictions[i]) for i in range(len(data_test))]
    scores = [sum([score[i] for score in scores]) for i in range(3)]
    precision, recall, f1 = calc_f(scores)

    results_f1.append(f1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        frame_ids = inputs.pop("frame_id")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # create labels mask based on frame ids
        frame_ids = [str(frame_id) for frame_id in frame_ids.tolist()]
        kept_labels = [frame2labelids[frame_id] for frame_id in frame_ids]
        mask = torch.zeros_like(logits)
        for i in range(logits.shape[0]):
            mask[i, :, kept_labels[i]] = 1
        # set logits of masked labels to ignore_index
        masked_logits = torch.where(mask.bool(), logits, torch.tensor(-100))
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(masked_logits.view(-1, self.model.config.num_labels), labels.view(-1))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train_model():
    with open('SRL_data_train_fulltext_sentanno.pickle', 'rb') as f:
        data_train = pickle.load(f)
    data_train = [sentanno for sentanno in data_train if sentanno.lu_id not in set(selected_verb_lus)]
    with open('SRL_data_val_sentanno.pickle', 'rb') as f:
        data_val = pickle.load(f)
    with open('SRL_data_test_sentanno.pickle', 'rb') as f:
        data_test = pickle.load(f)

    data_train, data_val, data_test = create_data(data_train), create_data(data_val), create_data(data_test)
    if args.local_rank == 0:
        print(len(data_train))
        print(len(data_val))
        print(len(data_test))

    dataset_dict_train = create_inputs_targets(data_train)
    dataset_dict_val = create_inputs_targets(data_val)
    dataset_dict_test = create_inputs_targets(data_test)

    ds_train = Dataset.from_dict(dataset_dict_train).with_format("torch")
    ds_val = Dataset.from_dict(dataset_dict_val).with_format("torch")
    ds_test = Dataset.from_dict(dataset_dict_test).with_format("torch")
    
    if args.local_rank == 0:
        print("dataset loaded")

    model = BertForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    if args.local_rank == 0:
        print("model loaded")

    model_name = model_checkpoint.split("/")[-1]
    train_args = TrainingArguments(
        f"{model_name}-FE-end2end",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy= "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "f1",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=50,
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # patience is 10 epochs
    early_stop = EarlyStoppingCallback(10)
    trainer = CustomTrainer(
        model,
        train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stop]
    )
    if args.local_rank == 0:
        # print(f"num labels: {model.config.num_labels}")
        print("start training...")
    trainer.train()
    trainer.save_model(f'spanbert-finetuned-srl-parser-fulltext')

    trainer.eval()
    preds_val = trainer.predict(ds_val).predictions
    results_val = compute_metrics((preds_val,dataset_dict_val['labels']))
    results_val_weighted = compute_metrics_weighted((preds_val,dataset_dict_val['labels']))

    model = BertForTokenClassification.from_pretrained('spanbert-finetuned-srl-parser-fulltext', num_labels=len(id2label), id2label=id2label, label2id=label2id)
    trainer = CustomTrainer(
        model,
        train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_test,
        callbacks=[early_stop]
    )
    preds_test = trainer.predict(ds_test).predictions
    results_test = compute_metrics_test((preds_test,dataset_dict_test['labels']))
    results_test_weighted = compute_metrics_test_weighted((preds_test,dataset_dict_test['labels']))
    results_test_core = compute_metrics_test_core((preds_test,dataset_dict_test['labels']))
    results_test_noncore = compute_metrics_test_noncore((preds_test,dataset_dict_test['labels']))
    
    results = {'val':results_val, 'val weighted':results_val_weighted, 'test':results_test,\
                'test weighted':results_test_weighted, 'test core':results_test_core,\
                'test noncore':results_test_noncore}
    if args.local_rank == 0:
        for key, val in results.items():
            print(f'{key}: {val}')
    return results

def train_model_aug(use_filter, tags, aug_ratio):
    with open('SRL_data_train_fulltext_sentanno.pickle', 'rb') as f:
        data_train = pickle.load(f)
    # with open('SRL_data_train_sentanno.pickle', 'rb') as f:
    #     data_train = pickle.load(f)

    if use_filter:
        with open(f'SRL_data_augmented_filtered_{tags}_{aug_ratio}_fulltext_v.pickle', "rb") as f:
            data_augmented = pickle.load(f)
    else:
        with open(f'SRL_data_augmented_{tags}_{aug_ratio}_fulltext_v.pickle', "rb") as f:
            data_augmented = pickle.load(f)
    data_train = data_train + data_augmented
    with open('SRL_data_val_sentanno.pickle', 'rb') as f:
        data_val = pickle.load(f)
    with open('SRL_data_test_sentanno.pickle', 'rb') as f:
        data_test = pickle.load(f)
    
    data_train, data_val, data_test = create_data(data_train), create_data(data_val), create_data(data_test)
    
    dataset_dict_train = create_inputs_targets(data_train)
    dataset_dict_val = create_inputs_targets(data_val)
    dataset_dict_test = create_inputs_targets(data_test)

    ds_train = Dataset.from_dict(dataset_dict_train).with_format("torch")
    ds_val = Dataset.from_dict(dataset_dict_val).with_format("torch")
    ds_test = Dataset.from_dict(dataset_dict_test).with_format("torch")
    
    if args.local_rank == 0:
        print("dataset loaded")

    model = BertForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    if args.local_rank == 0:
        print("model loaded")

    model_name = model_checkpoint.split("/")[-1]
    train_args = TrainingArguments(
        f"{model_name}-FE-end2end",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy= "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "f1",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=50,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # patience is 10 epochs
    early_stop = EarlyStoppingCallback(10)
    trainer = CustomTrainer(
        model,
        train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stop]
    )
    if args.local_rank == 0:
        # print(f"num labels: {model.config.num_labels}")
        print("start training...")
    trainer.train()
    trainer.save_model(f'spanbert-finetuned-srl-parser-fulltext-filter-{use_filter}-{tags}-{aug_ratio}')

    trainer.eval()
    preds_val = trainer.predict(ds_val).predictions
    results_val = compute_metrics((preds_val,dataset_dict_val['labels']))
    results_val_weighted = compute_metrics_weighted((preds_val,dataset_dict_val['labels']))
    
    model = BertForTokenClassification.from_pretrained(f'spanbert-finetuned-srl-parser-fulltext-filter-{use_filter}-{tags}-{aug_ratio}', num_labels=len(id2label), id2label=id2label, label2id=label2id)
    trainer = CustomTrainer(
        model,
        train_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_test,
        callbacks=[early_stop]
    )
    trainer.eval()
    preds_test = trainer.predict(ds_test).predictions
    results_test = compute_metrics_test((preds_test,dataset_dict_test['labels']))
    results_test_weighted = compute_metrics_test_weighted((preds_test,dataset_dict_test['labels']))
    results_test_core = compute_metrics_test_core((preds_test,dataset_dict_test['labels']))
    results_test_noncore = compute_metrics_test_noncore((preds_test,dataset_dict_test['labels']))

    results = {'val':results_val, 'val weighted':results_val_weighted, 'test':results_test,\
                'test weighted':results_test_weighted, 'test core':results_test_core,\
                'test noncore':results_test_noncore}
    if args.local_rank == 0:
        print(f'use_filter={use_filter}, tags={tags}, aug_ratio={aug_ratio}')
        for key, val in results.items():
            print(f'{key}: {val}')
    return results



if __name__ == "__main__":

    ########## TRAIN BASELINE MODEL ##########
    results_baseline = train_model()

    ########## TRAIN AUGMENTED MODEL ##########
    # results_aug = train_model_aug(False, 'no_tag', 0.25)
    # results_aug = train_model_aug(True, 'no_tag', 0.25)
    # results_aug = train_model_aug(False, 'FE_only', 0.25)
    results_aug = train_model_aug(True, 'FE_only', 0.25)
    # results_aug = train_model_aug(False, 'frame+FE', 0.25)
    # results_aug = train_model_aug(True, 'frame+FE', 0.25)
    # results_aug = train_model_aug(False, 'GPT_frame+FE', 0.25)
    results_aug = train_model_aug(True, 'GPT_frame+FE', 0.25)
    

