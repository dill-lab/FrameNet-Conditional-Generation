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
import math
import pandas as pd
import numpy as np
import xml.etree.ElementTree as et

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import spacy
import pyinflect
import re
import torch
from transformers import BertForSequenceClassification, BertModel, BertTokenizer, \
                         TrainingArguments, DataCollatorWithPadding, Trainer, \
                         LlamaForCausalLM, LlamaTokenizer
from datasets import load_metric, Dataset
import argparse
from sklearn.metrics import f1_score, accuracy_score

torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)

nlp = spacy.load("en_core_web_sm")
model_checkpoint = 'SpanBERT/spanbert-large-cased'
tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
max_len = 1024
batch_size = 16

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

class FrameNetDataset(Dataset):
    def __init__(self, input_ids, labels):
        # pad input_ids and attention_mask
        self.input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        self.attention_masks = torch.ones_like(self.input_ids)
        for i in range(len(input_ids)):
            self.attention_masks[i][len(input_ids[i]):] = 0
        self.labels = labels
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        attention_mask = self.attention_masks[index]
        label = self.labels[index]
        return input_ids, attention_mask, label

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

label2id = {label: i for i, label in enumerate(candidate_fes+['Not an FE'])}
id2label = {i: label for i, label in enumerate(candidate_fes+['Not an FE'])}
CLS = "[CLS]"
SEP = "[SEP]"
LU_START = "[unused1]"
LU_END = "[unused2]"
FE_START = "[unused3]"
FE_END = "[unused4]"

class TrainSentence:
    def __init__(self, frame, text, fe_start_ind, fe_end_ind, lu_start_ind, lu_end_ind, lu_id, label):
        self.label = label
        self.frame = frame
        self.text = text
        self.fe_start_ind = fe_start_ind
        self.fe_end_ind = fe_end_ind
        self.lu_start_ind = lu_start_ind
        self.lu_end_ind = lu_end_ind
        self.lu_id = lu_id
        self.skip = False

    def preprocess(self):
        label = self.label
        frame = self.frame
        text = self.text
        fe_start_ind = self.fe_start_ind
        fe_end_ind = self.fe_end_ind
        lu_start_ind = self.lu_start_ind
        lu_end_ind = self.lu_end_ind

        # truncate sentence
        text = text[:max_len]
        # check if FE span and LU span is in truncated sentence
        if fe_start_ind >= len(text) or fe_end_ind >= len(text) or lu_start_ind >= len(text) or lu_end_ind >= len(text):
            self.skip = True
            return

        tokens = [CLS]
        tokenized_ind = 0
        for ind in range(len(text)):
            if ind == fe_start_ind:
                tokens += tokenizer.tokenize(text[tokenized_ind:ind])
                tokens.append(FE_START)
                tokenized_ind = fe_start_ind
            elif ind == fe_end_ind:
                tokens += tokenizer.tokenize(text[tokenized_ind:(ind+1)])
                tokens.append(FE_END)
                tokenized_ind = fe_end_ind+1
            elif ind == lu_start_ind:
                tokens += tokenizer.tokenize(text[tokenized_ind:ind])
                tokens.append(LU_START)
                tokenized_ind = lu_start_ind
            elif ind == lu_end_ind:
                tokens += tokenizer.tokenize(text[tokenized_ind:(ind+1)])
                tokens.append(LU_END)
                tokenized_ind = lu_end_ind+1

        tokens += tokenizer.tokenize(text[tokenized_ind:])
        tokens.append(SEP)
        tokens += tokenizer.tokenize(frame)
        tokens.append(SEP)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_id = label2id[label]

        self.tokens = tokens
        self.input_ids = input_ids
        self.label_id = label_id

def create_data_one_sent(lu_id, sentanno):
    text = sentanno.text
    if len(sentanno.target_stinds) == 0:
        return []
    lu_start = sentanno.target_stinds[0]
    lu_end = sentanno.target_eninds[0]
    core_fes = [fe for fe in sentanno.fes if fe['is_core']]
    sents = []
    for fe in core_fes:
        try:
            # mask PP and FE type belonging to mask FE type
            if fe['phrase_type'] == "PP" or fe['id'] in candidate_fes:
                fe_start = fe['startind']
                fe_end = fe['endind']
                frame = id2frame[lu2frame[lu_id]]
                fe_id = fe['id']
                sent = TrainSentence(frame, text, fe_start, fe_end, lu_start, lu_end, lu_id, fe_id)
                sent.preprocess()
                sents.append(sent)
        except KeyError:
            pass
    return sents

def create_data_one_sent_empty(sentanno):
    lu_id = sentanno.lu_id
    text = sentanno.text
    lu_name = id2lu[lu_id]
    pos = lu_name[(lu_name.rfind('.')+1):]
    if pos != 'v':
        return [], []
    lu_start = sentanno.target_stinds[0]
    lu_end = sentanno.target_eninds[0]
    core_fes = [fe for fe in sentanno.fes if fe['is_core']]
    # print(len(core_fes))
    sents = []
    gold_fes = []
    for fe in core_fes:
        try:
            # mask PP and FE type belonging to mask FE type
            if fe['phrase_type'] == "PP" or fe['id'] in candidate_fes:
                fe_start = fe['startind']
                fe_end = fe['endind']
                frame = id2frame[lu2frame[lu_id]]
                fe_id = fe['id']
                sent = TrainSentence(frame, text, fe_start, fe_end, lu_start, lu_end, lu_id, fe_id)
                sent.preprocess()
                if sent.skip == False:
                    sents.append(sent)
                    gold_fes.append(fe_id)
        except KeyError:
            pass
    return gold_fes, sents

def create_data(data_sentanno):
    data = []
    if args.local_rank == 0:
        for sentanno in tqdm(data_sentanno):
            try:
                data += create_data_one_sent(sentanno.lu_id, sentanno)
            except KeyError:
                pass
    else:
        for sentanno in data_sentanno:
            try:
                data += create_data_one_sent(sentanno.lu_id, sentanno)
            except KeyError:
                pass

    return data

def create_data_empty(data_sentanno):
    data_empty = []
    gold_fes = []
    for sentanno in data_sentanno:
        gold_fes_per_sent, sents = create_data_one_sent_empty(sentanno)
        data_empty += sents 
        gold_fes.append(gold_fes_per_sent)
    return gold_fes, data_empty

def create_data_invalid_FE(data_sample):
    data_invalid = []
    # create data for label "Not an FE"
    for sent in data_sample:
        text = " " + sent.text
        # truncate sentence
        text = text[:(max_len+1)]
        lu_start = sent.lu_start_ind
        lu_end = sent.lu_end_ind
        neg_fe_start_end = random.sample(list(range(1, len(text)-1)), 2)
        neg_fe_start = min(neg_fe_start_end)
        neg_fe_end = max(neg_fe_start_end)
        # FE cannot overlap with LU
        # FE must be complete words (i.e. preceded by empty space and followed by empty space)
        while max(neg_fe_start, lu_start+1) < min(neg_fe_end, lu_end+1) or text[neg_fe_start-1] != " " or text[neg_fe_end+1] != " ":
            neg_fe_start_end = random.sample(list(range(1, len(text)-1)), 2)
            neg_fe_start = min(neg_fe_start_end)
            neg_fe_end = max(neg_fe_start_end)
        sent = TrainSentence(sent.frame, text[1:], neg_fe_start-1, neg_fe_end-1, lu_start, lu_end, sent.lu_id, 'Not an FE')
        sent.preprocess()
        data_invalid.append(sent)
    return data_invalid

def create_inputs_targets(data):
    dataset_dict = {
        "input_ids": [],
        "label_id": [],
    }
    for item in data:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    dataset_dict["labels"] = dataset_dict.pop("label_id")
    return dataset_dict

def acc_and_f1(eval_pred):
    preds, labels = eval_pred
    preds = preds.argmax(axis=1)
    acc = accuracy_score(y_true=labels, y_pred=preds).item()
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro').item()
    return {
        "accuracy": acc,
        "f1": f1,
    }

def get_generation_input(empty_id, sentanno):
    inp = sentanno.text
    target_replacement = id2lu[empty_id]
    # discard tag
    target_replacement = target_replacement[:(target_replacement.rfind("."))]
    # discard non-alphanumeric, keep space, hyphen, and apostrophe
    target_replacement = re.sub(r'[^a-zA-Z0-9 -\']', '', target_replacement)
    target_to_replace = inp[sentanno.target_stinds[0]:(sentanno.target_eninds[0]+1)]
    # handle inflection inconsistency
    target_replacement = inflect_replacement(target_to_replace, target_replacement)
    
    # # for original sentence, target replacement is the same as target to replace
    # target_replacement = inp[sentanno.target_stinds[0]:(sentanno.target_eninds[0]+1)]
    
    # if not verb LU, only perform replacement
    lu_name = id2lu[empty_id]
    pos = lu_name[(lu_name.rfind('.')+1):]
    if pos != 'v':
        inp = inp[:sentanno.target_stinds[0]] + target_replacement + inp[(sentanno.target_eninds[0]+1):]
        return inp
    # replacement only
    # inp = inp[:sentanno.target_stinds[0]] + target_replacement + inp[(sentanno.target_eninds[0]+1):]
    # return inp

    core_fes = [fe for fe in sentanno.fes if fe['is_core']]
    new_target = {'startind':sentanno.target_stinds[0], 'endind':sentanno.target_eninds[0]+1,\
                          'text':target_replacement}
    core_fes.append(new_target)
    # sort FEs by start index in descending order
    core_fes = sorted(core_fes, key=lambda d: d['startind'], reverse=True)
    mask_id = 0
    mask_fes = []
    frame = None
    for fe in core_fes:
        # is target
        if fe['startind'] == sentanno.target_stinds[0]:
            # replace target in sentence
            inp = inp[:fe['startind']] + new_target['text'] + inp[fe['endind']:]
        else:
            try:
                # mask PP and FE type belonging to mask FE type
                if fe['phrase_type'] == "PP" or fe['id'] in candidate_fes:
                    input_cp = inp
                    start = fe['startind']
                    end = fe['endind']
                    masked_fe = input_cp[start:(end+1)]
                    frame = id2frame[lu2frame[empty_id]]
                    fe_type = fe['name']
                    mask_fes.append(fe_type)
                    mask_text = f"<extra_id_{mask_id}>"
                    # mask FE in input with mask text
                    inp = input_cp[:start] + mask_text + input_cp[(end+1):]
                    mask_id += 1
            except KeyError:
                pass
    return inp

def get_tokens_list(tags, input, preds):
    if len(preds) == 0:
        return []
    mask_id = 0
    output = []
    sequence = preds
    # T5 models generation
    if tags.find('GPT') == -1:
        while (input.find(f"<extra_id_{mask_id}>") != -1):
            token_len = len(f"<extra_id_{mask_id}>")
            tok_start = sequence.find(f"<extra_id_{mask_id}>") + token_len + 1
            tok_end = sequence.find(f"<extra_id_{mask_id+1}>")
            if tok_start >= token_len and tok_start < len(sequence) and tok_end != -1:
                # append pred span
                output.append(sequence[tok_start:tok_end])
            else:
                output.append("")
            mask_id += 1
    # GPT models generation
    else:
        output = sequence.split(', ')
        # truncate preds to size of mask tokens
        num_masks = len(re.findall(r'<extra_id_[0-9]+>', input))
        output = output[:num_masks]
        # pad preds to size of mask tokens
        output = output + [''] * (num_masks - len(output))
        output.reverse()

    return output

def get_fe_indices(data, input, preds, target_replacement):
    empty_id, sentanno = data
    lu_name = id2lu[empty_id]
    pos = lu_name[(lu_name.rfind('.')+1):]
    # in case FE is at beginning of sentence
    sent = " " + input
    sent_orig = sentanno.text
    # FEs sorted w.r.t. start inds in ascending order
    fes = sorted(sentanno.fes, key=lambda d: d['startind'])
    search_start = 0
    mask_id = len(re.findall(r'<extra_id_[0-9]+>', sent)) - 1
    new_sentanno = SentenceFEs(empty_id, sentanno.id+'n', sent, sentanno.core_fes)
    for fe in fes:
        try:
            # masked FE
            if pos == 'v' and mask_id >= 0 and fe['is_core'] and (fe['phrase_type'] == "PP" or fe['id'] in mask_FE):
                fe_start = sent.find(f"<extra_id_{mask_id}>") - 1
                if mask_id < len(preds):
                    # print(mask_id, len(preds))
                    sent = sent.replace(f"<extra_id_{mask_id}>", preds[mask_id])
                    fe_end = fe_start + len(preds[mask_id])
                else:
                    # if masked FE not in preds, replace with empty string
                    sent = sent.replace(f"<extra_id_{mask_id}>", "")
                    fe_end = fe_start
                new_sentanno.add_fe(id2fe[fe['id']], fe['id'], fe_start, fe_end-1)
                new_sentanno.add_fe_phrasetype(fe_start, fe['phrase_type'])
                mask_id -= 1
            # non-masked FE
            else:
                fe_span = sent_orig[fe['startind']:(fe['endind']+1)]
                fe_start = sent.find(f' {fe_span} ', search_start)
                fe_end = fe_start + len(fe_span)
                search_start = fe_end
                new_sentanno.add_fe(id2fe[fe['id']], fe['id'], fe_start, fe_end-1)
        # non-masked FE
        except KeyError:
            # print('non-masked FE')
            fe_span = sent_orig[fe['startind']:(fe['endind']+1)]
            fe_start = sent.find(f' {fe_span} ', search_start)
            fe_end = fe_start + len(fe_span)
            search_start = fe_end
            new_sentanno.add_fe(id2fe[fe['id']], fe['id'], fe_start, fe_end-1)

    # record LU inds
    lu_start = sent.find(f' {target_replacement} ')
    lu_end = lu_start + len(target_replacement)
    new_sentanno.text = sent[1:]
    new_sentanno.add_target(lu_start, lu_end-1)

    return new_sentanno

def inflect_replacement(target_to_replace, target_replacement):
    # if target contains multiple words, then use tag of verb, otherwise first word
    tag = None
    for token in nlp(target_to_replace):
        if 'VB' in token.tag_:
            tag = token.tag_ 
            break
    if tag is None:
        tag = nlp(target_to_replace)[0].tag_
    replacement_token = None
    for token in nlp(target_replacement):
        if 'VB' in token.tag_:
            replacement_token = token
            break
    if replacement_token is None:
        replacement_token = nlp(target_replacement)[0]
    inflected = replacement_token._.inflect(tag)
    if inflected is not None:
        target_replacement = target_replacement.replace(replacement_token.text, inflected)
    return target_replacement

def get_target_replacement(empty_id, sentanno):
    target_replacement = id2lu[empty_id]
    # discard tag
    target_replacement = target_replacement[:(target_replacement.rfind("."))]
    # discard non-alphanumeric, keep space, hyphen, and apostrophe
    target_replacement = re.sub(r'[^a-zA-Z0-9 -\']', '', target_replacement)
    target_to_replace = sentanno.text[sentanno.target_stinds[0]:(sentanno.target_eninds[0]+1)]
    # handle inflection inconsistency
    target_replacement = inflect_replacement(target_to_replace, target_replacement)
    return target_replacement

def get_orig_fe_spans(sentanno):
    text = sentanno.text
    core_fes = [fe for fe in sentanno.fes if fe['is_core']]
    # sort FEs by start index in descending order
    core_fes = sorted(core_fes, key=lambda d: d['startind'], reverse=True)
    mask_id = 0
    output = ""
    for fe in core_fes:
        try:
            # mask PP and FE type belonging to mask FE type
            if fe['phrase_type'] == "PP" or fe['id'] in mask_FE:
                start = fe['startind']
                end = fe['endind']
                masked_fe = text[start:(end+1)]
                output += f"<extra_id_{mask_id}> {masked_fe} "
                mask_id += 1
        except KeyError:
            pass
    output += f"<extra_id_{mask_id}>"

    return output


def prepare_augmented_data(tags, aug_ratio, num_candidates=1):
    # with open('SRL_data_train_sentanno.pickle', 'rb') as f:
    #     data_train_srl = pickle.load(f)
    with open('SRL_data_train_fulltext_sentanno.pickle', 'rb') as f:
        data_train_srl = pickle.load(f)
    with open(f'SRL_data_to_augment.pickle', "rb") as f:
        data_to_augment = pickle.load(f)
    num_aug = min(math.ceil(aug_ratio*len(data_train_srl)), len(data_to_augment))
    # augment train data
    data_to_augment = data_to_augment[:num_aug]
    # duplicate each element 3 times
    data_to_augment = [element for element in data_to_augment for _ in range(num_candidates)]

    if tags == 'orig':
        y_pred = [get_orig_fe_spans(sentanno) for empty_id, sentanno in data_to_augment]
    else:
        with open(f'y_pred_empty_{tags}_fulltext_v.pickle', 'rb') as f:
            y_pred = pickle.load(f)
    target_replacement = [get_target_replacement(*data) for data in data_to_augment]
    
    # input sentence to T5
    X_gen = [get_generation_input(*data) for data in data_to_augment]
    # convert to list-of-lists format
    y_pred = [get_tokens_list(tags, X_gen[i], y_pred[i]) for i in range(len(data_to_augment))]

    X_out = []
    fe_start_end_inds = []
    lu_start_end_inds = []
    data_augmented = []
    for i in range(len(data_to_augment)):
        # sentanno, input, preds, target_replacement
        new_sentanno = get_fe_indices(data_to_augment[i], X_gen[i], y_pred[i], target_replacement[i])
        # instances with new FE spans
        if len(y_pred[i]) > 0:
            data_augmented.append(new_sentanno)

    return data_augmented

def compute_perplexity(texts):
    # model_id = "gpt2-large"
    model_id = "meta-llama/Llama-2-7b-hf"
    # model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model = LlamaForCausalLM.from_pretrained(model_id, use_auth_token='INSERT_TOKEN_HERE').to(device)
    # tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    tokenizer = LlamaTokenizer.from_pretrained(model_id, use_auth_token='INSERT_TOKEN_HERE')
    ppls = []

    for text in tqdm(texts):
        encodings = tokenizer(text, return_tensors="pt")
        max_length = model.config.max_length
        stride = 8
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        ppls.append(ppl)

    return sum(ppls) / len(ppls)

def train_model():
    with open('SRL_data_train_sentanno.pickle', 'rb') as f:
        data_train = pickle.load(f)
    with open('SRL_data_val_sentanno.pickle', 'rb') as f:
        data_val = pickle.load(f)
    with open('SRL_data_test_sentanno.pickle', 'rb') as f:
        data_test = pickle.load(f)

    data_train, data_val_complete, data_test_complete = create_data(data_train), create_data(data_val), create_data(data_test)
    data_train_sample = random.sample(data_train, math.ceil(len(data_train)*0.0175))
    data_train_invalid = create_data_invalid_FE(data_train_sample)
    data_train_complete = data_train + data_train_invalid

    dataset_dict_train = create_inputs_targets(data_train_complete)
    dataset_dict_val = create_inputs_targets(data_val_complete)
    dataset_dict_test = create_inputs_targets(data_test_complete)

    ds_train = Dataset.from_dict(dataset_dict_train).with_format("torch")
    ds_val = Dataset.from_dict(dataset_dict_val).with_format("torch")
    ds_test = Dataset.from_dict(dataset_dict_test).with_format("torch")
    if args.local_rank == 0:
        print("dataset loaded")

    model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    # model = BertForSequenceClassification.from_pretrained('spanbert-finetuned-fe-classifier', num_labels=len(id2label), id2label=id2label, label2id=label2id)
    if args.local_rank == 0:
        print("model loaded")

    model_name = model_checkpoint.split("/")[-1]
    train_args = TrainingArguments(
        f"{model_name}-FE-classifier",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy= "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "accuracy",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=20,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model,
        train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=acc_and_f1,
    )
    if args.local_rank == 0:
        # print(f"num labels: {model.config.num_labels}")
        print("start training...")
    trainer.train()
    trainer.save_model('spanbert-finetuned-fe-classifier')
    
    preds_val = trainer.predict(ds_val).predictions
    results_val = acc_and_f1((preds_val,dataset_dict_val['labels']))
    preds_test = trainer.predict(ds_test).predictions
    results_test = acc_and_f1((preds_test,dataset_dict_test['labels']))

    if args.local_rank == 0:
        print('val:', results_val)
        print('test:', results_test)
    
def test_empty_lus(tags, aug_ratio):
    if args.local_rank == 0:
        print(tags, aug_ratio)

    data_augmented = prepare_augmented_data(tags, aug_ratio)
    with open(f'SRL_data_augmented_{tags}_{aug_ratio}_fulltext_v.pickle', "wb") as f:
        pickle.dump(data_augmented, f)

    gold_fes, data_sents_aug = create_data_empty(data_augmented)
    gold_fes_flat = [fe for gold_fes_per_sent in gold_fes for fe in gold_fes_per_sent]
    dataset_dict_aug = create_inputs_targets(data_sents_aug)
    ds_aug = Dataset.from_dict(dataset_dict_aug).with_format("torch")
    model = BertForSequenceClassification.from_pretrained('spanbert-finetuned-fe-classifier')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    preds_aug_logits = trainer.predict(ds_aug).predictions
    results_aug = acc_and_f1((preds_aug_logits,dataset_dict_aug['labels']))
    if args.local_rank == 0:
        print(results_aug)
    preds_aug = preds_aug_logits.argmax(axis=1)
    pred_fes_aug = [id2label[pred] for pred in preds_aug.tolist()]
    data_aug_filtered = []
    # pointer to pred_fes_aug 
    ind = 0
    for i in range(len(gold_fes)):
        gold_fes_per_sent = gold_fes[i]
        j = 0
        filter_passed = True
        # iterate through gold fes for current sentence
        # cannot pass filter when any gold fe != pred fe
        for fe in gold_fes_per_sent:
            if fe != pred_fes_aug[ind]:
                filter_passed = False
            ind += 1
        if filter_passed:
            # all gold fes == pred fes
            data_aug_filtered.append(data_augmented[i])

    with open(f'SRL_data_augmented_filtered_{tags}_{aug_ratio}_fulltext_v.pickle', "wb") as f:
        pickle.dump(data_aug_filtered, f)
    if args.local_rank == 0:
        print('num total:', len(data_augmented))
        print('num kept:', len(data_aug_filtered))

    return


if __name__ == "__main__":

    ########## TRAIN ##########
    train_model()

    ########## FILTER INCONSISTENT FES ##########
    test_empty_lus('no_tag', 0.25)
    test_empty_lus('FE_only', 0.25)
    test_empty_lus('frame+FE', 0.25)
    


