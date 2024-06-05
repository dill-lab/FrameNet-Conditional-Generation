from __future__ import division
import json
import pickle
import sys
import time

'''
Reads XML files containing FrameNet 1.$VERSION annotations, and converts them to a CoNLL 2009-like format.
'''
import codecs
import os

import importlib
importlib.reload(sys)

from tqdm.auto import tqdm
import random
import math
import pandas as pd
import numpy as np
import xml.etree.ElementTree as et

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

import openai
import spacy
import pyinflect
import re
from collections import Counter
import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5Tokenizer, T5ForConditionalGeneration
from datasets import load_metric
from datasets import Dataset
from BARTScore.bart_score import BARTScorer
import argparse

torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)
local_rank = args.local_rank

# print("Number of available GPUs:", torch.cuda.device_count())
# print("Local rank:", args.local_rank)

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

class TrainSentence:
    def __init__(self, sent_id, frame, text, fe_start_inds, fe_end_inds, lu_start_ind, lu_end_ind, lu_id, fe_ids):
        self.frame = frame
        self.text = text
        self.fe_start_inds = fe_start_inds
        self.fe_end_inds = fe_end_inds
        self.lu_start_ind = lu_start_ind
        self.lu_end_ind = lu_end_ind
        self.lu_id = lu_id
        self.fe_ids = fe_ids
        self.skip = False

    def preprocess(self):
        pass

batch_size = 16
max_input_length = 1024
max_target_length = 1024

model_checkpoint = "t5-large"
device = "cuda"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
bart_scorer = BARTScorer(device="cuda", checkpoint="facebook/bart-large")
nlp = spacy.load("en_core_web_sm")
openai.api_key = 'INSERT_API_KEY_HERE'

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

def prepare_data(data, tags, lu_id, sentanno):
    # if not verb, skip
    lu = id2lu[lu_id]
    pos = lu[(lu.rfind('.')+1):]
    if pos != 'v':
        return

    input = sentanno.text
    output = ""
    core_fes = [fe for fe in sentanno.fes if fe['is_core']]
    # sort FEs by start index in descending order
    core_fes = sorted(core_fes, key=lambda d: d['startind'], reverse=True)
    # keep track of whether masking happened
    mask_id = 0
    for fe in core_fes:
        try:
            # mask PP and FE type belonging to mask FE type
            if fe['phrase_type'] == "PP" or fe['id'] in candidate_fes:
                mask = True
                input_cp = input
                output_cp = output
                start = fe['startind']
                end = fe['endind']
                masked_fe = input_cp[start:(end+1)]
                frame = id2frame[lu2frame[lu_id]]
                fe_type = fe['name']
                if tags == 'frame+FE':
                    mask_text = f"<Frame:{frame}+FE:{fe_type}> <extra_id_{mask_id}> </Frame:{frame}+FE:{fe_type}>"
                elif tags == 'FE_only':
                    mask_text = f"<FE:{fe_type}> <extra_id_{mask_id}> </FE:{fe_type}>"
                elif tags == 'no_tag' or tags.find('zero_shot') != -1:
                    mask_text = f"<extra_id_{mask_id}>"
 
                # mask FE in input with mask text
                input = input_cp[:start] + mask_text + input_cp[(end+1):]
                output += f"<extra_id_{mask_id}> {masked_fe} "
                mask_id += 1
        except KeyError:
            pass
    # no FE is masked in this sentence
    if mask_id == 0:
        return

    data['X'].append(input)
    data['y'].append(output+f"<extra_id_{mask_id}>")
    return

    
def prepare_data_gpt(data, tags, lu_id, sentanno):
    # if not verb, skip
    lu = id2lu[lu_id]
    pos = lu[(lu.rfind('.')+1):]
    if pos != 'v':
        return

    input = sentanno.text
    output = ""
    core_fes = [fe for fe in sentanno.fes if fe['is_core']]
    # sort FEs by start index in descending order
    core_fes = sorted(core_fes, key=lambda d: d['startind'], reverse=True)
    # keep track of whether masking happened
    mask = False
    mask_fe_types = []
    for fe in core_fes:
        try:
            # mask PP and FE type belonging to mask FE type
            if fe['phrase_type'] == "PP" or fe['id'] in candidate_fes:
                mask = True
                input_cp = input
                output_cp = output
                start = fe['startind']
                end = fe['endind']
                masked_fe = input_cp[start:(end+1)]
                frame = id2frame[lu2frame[lu_id]]
                fe_type = fe['name']
                mask_text = "<mask>"
                # insert current fe to beginning of fe list
                mask_fe_types = [fe_type] + mask_fe_types
                # mask FE in input with mask text
                input = input_cp[:start] + mask_text + input_cp[(end+1):]
                output = f"{masked_fe}, " + output

        except KeyError:
            pass
    # no FE is masked in this sentence
    if not mask:
        return
    # GPT models
    else:
        # GPT prompts
        if tags.find('GPT_no_tag') != -1:
            input_prompt = f'Sentence: {input}'
        elif tags.find('GPT_FE_only') != -1:
            input_prompt = f"Lexical Unit: {id2lu[lu_id]}. Sentence: {input}. FE Type: "
            for i in range(len(mask_fe_types)):
                if i < (len(mask_fe_types)-1):
                    input_prompt += f'{mask_fe_types[i]}, '
                else:
                    input_prompt += f'{mask_fe_types[i]}. '
        elif tags.find('GPT_frame+FE') != -1:
            input_prompt = f"Frame: {id2frame[lu2frame[lu_id]]}. Lexical Unit: {id2lu[lu_id]}. Sentence: {input}. FE Type: "
            for i in range(len(mask_fe_types)):
                if i < (len(mask_fe_types)-1):
                    input_prompt += f'{mask_fe_types[i]}, '
                else:
                    input_prompt += f'{mask_fe_types[i]}. '

    data['X'].append(input_prompt)
    data['y'].append(output[:-2])
    return
    

def prepare_data_empty(tags, empty_id, nonempty_id, sentanno):
    item = {}
    input = sentanno.text
    target_replacement = id2lu[empty_id]
    # discard tag
    target_replacement = target_replacement[:(target_replacement.rfind("."))]
    # discard non-alphanumeric, keep space, hyphen, and apostrophe
    target_replacement = re.sub(r'[^a-zA-Z0-9 -\']', '', target_replacement)
    if len(sentanno.target_stinds) == 0:
        return
    target_to_replace = input[sentanno.target_stinds[0]:(sentanno.target_eninds[0]+1)]
    # handle inflection inconsistency
    target_replacement = inflect_replacement(target_to_replace, target_replacement)
    core_fes = [fe for fe in sentanno.fes if fe['is_core']]
    target_replacement = {'startind':sentanno.target_stinds[0], 'endind':sentanno.target_eninds[-1]+1,\
                          'text':target_replacement}
    core_fes.append(target_replacement)
    # sort FEs by start index in descending order
    core_fes = sorted(core_fes, key=lambda d: d['startind'], reverse=True)
    mask_id = 0
    mask_fes = []
    frame = None
    for fe in core_fes:
        # is target
        if fe['startind'] == sentanno.target_stinds[0]:
            # replace target in sentence
            input = input[:fe['startind']] + target_replacement['text'] + input[fe['endind']:]
        else:
            try:
                # mask PP and FE type belonging to mask FE type
                if fe['phrase_type'] == "PP" or fe['id'] in candidate_fes:
                    mask = True
                    input_cp = input
                    start = fe['startind']
                    end = fe['endind']
                    masked_fe = input_cp[start:(end+1)]
                    frame = id2frame[lu2frame[nonempty_id]]
                    fe_type = fe['name']
                    mask_fes.append(fe_type)
                    if tags == 'frame+FE':
                        mask_text = f"<Frame:{frame}+FE:{fe_type}> <extra_id_{mask_id}> </Frame:{frame}+FE:{fe_type}>"
                    elif tags == 'FE_only':
                        mask_text = f"<FE:{fe_type}> <extra_id_{mask_id}> </FE:{fe_type}>"
                    elif tags == 'no_tag' or tags.find('zero_shot') != -1:
                        mask_text = f"<extra_id_{mask_id}>"
                    # GPT 
                    else:
                        mask_text = f"<mask>"
                    # mask FE in input with mask text
                    input = input_cp[:start] + mask_text + input_cp[(end+1):]
                    mask_id += 1
            except KeyError:
                pass

    if tags.find('GPT') != -1:
        # reverse order of GPT generations
        mask_fes.reverse()
        # GPT prompts
        if tags.find('GPT_no_tag') != -1:
            input_prompt = f'Sentence: {input}'
        elif tags.find('GPT_FE_only') != -1:
            input_prompt = f"Lexical Unit: {id2lu[empty_id]}. Sentence: {input}. FE Type: "
            for i in range(len(mask_fes)):
                if i < (len(mask_fes)-1):
                    input_prompt += f'{mask_fes[i]}, '
                else:
                    input_prompt += f'{mask_fes[i]}. '
        elif tags.find('GPT_frame+FE') != -1:
            input_prompt = f"Frame: {frame}. Lexical Unit: {id2lu[empty_id]}. Sentence: {input}. FE Type: "
            for i in range(len(mask_fes)):
                if i < (len(mask_fes)-1):
                    input_prompt += f'{mask_fes[i]}, '
                else:
                    input_prompt += f'{mask_fes[i]}. '
        item['X_prompt'] = input_prompt
    item['X'] = input
    item['target_replacement'] =  target_replacement['text']
    item['empty_id'] = empty_id
    item['nonempty_id'] = nonempty_id
    item['sent_id'] = sentanno.id

    return item

def tokenize(data):
    model_inputs = tokenizer(data['X'], max_length=max_input_length, truncation=True)

    # data may or may not have labels
    try:
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(data['y'], max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
    except KeyError:
        pass

    return model_inputs

def get_tokens_list(labels, preds, to_list):
    mask_id = 0
    output = []
    if (to_list == "labels"):
        sequence = labels 
    else:
        sequence = preds
    while (labels.find(f"<extra_id_{mask_id+1}>") != -1):
        token_len = len(f"<extra_id_{mask_id}>")
        tok_start = sequence.find(f"<extra_id_{mask_id}>") + token_len
        tok_end = sequence.find(f"<extra_id_{mask_id+1}>")
        if tok_start >= token_len and tok_start < len(sequence) and tok_end != -1:
            # append pred span
            output.append(sequence[tok_start:tok_end])
        else:
            output.append("")
        mask_id += 1
    return output

def get_tokens_list_gpt(labels, preds):
    output = []
    labels_list = labels.split(', ')
    preds_list = preds.split(', ')
    # truncate preds to labels size
    output = preds_list[:len(labels_list)]
    # pad preds to labels size
    output = output + [''] * (len(labels_list) - len(output))

    return output

def get_output_sentence(input, preds):
    mask_id = 0
    output = input
    while (output.find(f"<extra_id_{mask_id}>") != -1):
        token_len = len(f"<extra_id_{mask_id}>")
        pred_start = preds.find(f"<extra_id_{mask_id}>") + token_len
        pred_end = preds.find(f"<extra_id_{mask_id+1}>")
        if pred_start >= token_len and pred_start < len(preds):
            if pred_end != -1:
                # replace mask token with pred
                output = output.replace(f"<extra_id_{mask_id}>", preds[pred_start:pred_end])
            else:
                output = output.replace(f"<extra_id_{mask_id}>", preds[pred_start:])
        mask_id += 1
    return output

def get_output_sentence_gpt(input, preds):
    output = input
    # truncate preds to size of mask tokens
    preds = preds[:input.count("<mask>")]
    for pred in preds:
        # replace mask token with pred
        output = output.replace("<mask>", pred, 1)
    return output

def remove_tags(tags, text):
    # remove <unk> token
    text = re.sub(r'\<unk\>', '', text)
    if tags == 'frame+FE' or tags == 'FE_only':
        text = re.sub(r'\<.*?FE.*?\>', '', text)
    # remove multiple empty spaces
    text = re.sub(r' +', ' ', text)
    return text

def compute_bart_score(labels, preds):
    if len(labels) != len(preds) or len(labels) == 0:
        return None
    # print(labels)
    # print(preds)
    precision = np.exp(np.array(bart_scorer.score(labels, preds, batch_size=batch_size)))
    recall = np.exp(np.array(bart_scorer.score(preds, labels, batch_size=batch_size)))
    f1 = (2 * precision * recall) / (precision + recall)
    # print(np.average(f1))
    return np.average(f1)

def compute_rouge(labels, preds):
    metric = load_metric("rouge")
    result = metric.compute(predictions=preds, references=labels)
    result = {key: value.mid.fmeasure for key, value in result.items()}
    return result

def fill_mask(data, model):
    batch = tokenizer(data, max_length=max_input_length, truncation=True, padding=True, return_tensors='pt')
    batch = batch.to(device)
    generated_ids = model.generate(batch["input_ids"], max_length=max_target_length)
    output = tokenizer.batch_decode(generated_ids)[0]
    return output

def fill_mask_empty(data, model):
    # no masked FE in sentence
    if data.find("extra_id") == -1:
        return ''
    inp = tokenizer(data, max_length=max_input_length, truncation=True, padding=True, return_tensors='pt')
    inp = inp.to(device)
    generated_ids = model.generate(inp["input_ids"], max_length=max_target_length)
    output = tokenizer.batch_decode(generated_ids)[0]
    return output

def gpt_generate(prompt):
    openai_model = "gpt-4"
    resp = openai.ChatCompletion.create(
        model = openai_model,
        messages = [
            {'role': 'user', 'content': prompt}
          ],
        temperature = 0,
        max_tokens = 1024
    )
    output = resp['choices'][0]['message']['content']
    return output

def create_inputs_targets(data):
    dataset_dict = {
        "X": [],
        "y": [],
    }
    for item in data:
        for key in dataset_dict:
            dataset_dict[key].append(item[key])
    return dataset_dict

def create_inputs_targets_empty(data):
    dataset_dict = {
        "X": [],
    }
    for item in data:
        for key in dataset_dict:
            dataset_dict[key].append(item[key])
    return dataset_dict

def get_verb_only_dataset(data):
    print(f'before: {len(data)}')
    data_v = []
    for sentanno in data:
        lu_name = id2lu[sentanno.lu_id]
        pos_tag = lu_name[lu_name.rfind('.')+1:]
        if pos_tag == 'v':
            data_v.append(sentanno)
    print(f'after: {len(data_v)}')
    return data_v

def prepare_data_to_augment():
    with open('SRL_data_train_fulltext_sentanno.pickle', "rb") as f:
        data_train_srl = pickle.load(f)
    with open('SRL_data_val_sentanno.pickle', "rb") as f:
        data_val_srl = pickle.load(f)
    with open('SRL_data_test_sentanno.pickle', "rb") as f:
        data_test_srl = pickle.load(f)

    data_empty = []
    num_augmented = len(data_train_srl)
    lus_train = [item.lu_id for item in data_train_srl]
    lu_counts_train = Counter(lus_train)
    lus_val_test = [item.lu_id for item in data_val_srl+data_test_srl]
    lu_counts_val_test = Counter(lus_val_test)
    lus_test = [item.lu_id for item in data_test_srl]
    lu_counts_test = Counter(lus_test)
    
    # augment LUs in val&test that appears rarely in train data
    lus_to_augment = list(set(lus_val_test))
    print(len(set(lus_val_test)))
    print(len(lus_to_augment))

    sents_train = set([item.id for item in data_train_srl])
    # importance score of an LU is computed as
    # freq fraction in val&test / freq fraction in train
    # alternate strategy: augment most common LUs in test
    def compute_importance(lu_id):
        train_imp = (lu_counts_train[lu_id]+1) / len(lus_train)
        val_test_imp = lu_counts_val_test[lu_id] / len(lus_val_test)
        imp = val_test_imp / train_imp
        return imp
    lus_importance = dict([(lu, compute_importance(lu)) for lu in lus_to_augment])
    # normalize importance scores
    imp_sum = sum(lus_importance.values())
    lus_imp_normalized = dict([(key,val/imp_sum) for key,val in lus_importance.items()])
    lu_counts_val_test = sorted(lu_counts_val_test.items(), key=lambda item: -item[1])
    # augmentation size for one LU is 
    # total augmentation size * normalized importance score
    lus_aug_size = dict([(key,math.ceil(num_augmented*val)) for key,val in lus_imp_normalized.items()])

    # print(len(lu_sent_counts))
    for lu_id, aug_size in tqdm(lus_aug_size.items()):
        empty_id = lu_id
        empty_pos = id2lu[empty_id][(id2lu[empty_id].rfind(".")+1):]
        # determine sister LU
        # sisters have structure [(id,has_sents), (id,has_sents), ...]
        sisters = frame2lus[lu2frame[empty_id]]
        used_sents = set()
        # augment current LU aug_size times
        for _ in range(aug_size):
            random.shuffle(sisters)
            for sister in sisters:
                sister_name = id2lu[sister]
                sister_pos = sister_name[(sister_name.rfind(".")+1):]
                # find a sister LU that's not empty and has same POS
                if sister != empty_id and sister in lu2sents and empty_pos == sister_pos:
                    # determine sentence for augmentation
                    sent_aug = None
                    sents = lu2sents[sister]
                    random.shuffle(sents)    
                    for sent in sents:
                        if sent.id in sents_train and sent.id not in used_sents:
                            sent_aug = sent
                            used_sents.add(sent.id)
                            data_empty.append((empty_id, sent_aug))
                            break
                    if sent_aug is not None:
                        break
    random.shuffle(data_empty)
    data_empty = [d for d in data_empty if len(d[1].target_stinds) > 0]
    with open(f'SRL_data_to_augment.pickle', 'wb') as f:
        pickle.dump(data_empty, f)
    return

def train_model(tags):
    data_raw_sample_train = {'X':[], 'y':[]}
    data_raw_sample_val = {'X':[], 'y':[]}
    data_raw_sample_test = {'X':[], 'y':[]}
    with open('SRL_data_train_sentanno.pickle', "rb") as f:
        data_train_srl = pickle.load(f)
    with open('SRL_data_val_sentanno.pickle', "rb") as f:
        data_val_srl = pickle.load(f)
    with open('SRL_data_test_sentanno.pickle', "rb") as f:
        data_test_srl = pickle.load(f)
    for sentanno in data_train_srl:
        try:
            prepare_data(data_raw_sample_train, tags, sentanno.lu_id, sentanno)
        except KeyError:
            pass
    for sentanno in data_val_srl:
        try:
            prepare_data(data_raw_sample_val, tags, sentanno.lu_id, sentanno)
        except KeyError:
            pass
    for sentanno in data_test_srl:
        try:
            prepare_data(data_raw_sample_test, tags, sentanno.lu_id, sentanno)
        except KeyError:
            pass

    data_tok_sample_train = tokenize(data_raw_sample_train)
    data_tok_sample_val = tokenize(data_raw_sample_val)
    data_tok_sample_test = tokenize(data_raw_sample_test)

    ds_train = Dataset.from_dict(data_tok_sample_train).with_format("torch")
    ds_val = Dataset.from_dict(data_tok_sample_val).with_format("torch")
    ds_test = Dataset.from_dict(data_tok_sample_test).with_format("torch")
    if args.local_rank == 0:
        print("dataset loaded")

    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    # model = T5ForConditionalGeneration.from_pretrained('t5-finetuned-framenet-'+tags+'-complete')
    if args.local_rank == 0:
        print("model loaded")
    model_name = model_checkpoint.split("/")[-1]
    train_args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-framenet",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy= "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "BARTScore_f1",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=5,
        predict_with_generate=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels)

        decoded_preds = [get_tokens_list(decoded_labels[i], decoded_preds[i], "preds") for i in range(len(decoded_preds))]
        decoded_labels = [get_tokens_list(decoded_labels[i], decoded_preds[i], "labels") for i in range(len(decoded_labels))]
        # flatten preds & labels
        decoded_preds = [y for sent in decoded_preds for y in sent]
        decoded_labels = [y for sent in decoded_labels for y in sent]
        rouge_scores = compute_rouge(decoded_labels, decoded_preds)
        bart_score = compute_bart_score(decoded_labels, decoded_preds)
        result = rouge_scores
        result['BARTScore_f1'] = bart_score

        return result

    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    if args.local_rank == 0:
        print("start training...")
    trainer.train()
    trainer.save_model('t5-finetuned-framenet-'+tags+'-complete')

    preds_test = trainer.predict(ds_test)
    results_test = preds_test.metrics
    print(tags)
    print(results_test)
    return

def predict_empty_batch(data_empty, model, num_candidates):
    # turn list of dicts to a dict with list as values
    data_empty_dict = create_inputs_targets_empty(data_empty)
    data_empty_tok = tokenize(data_empty_dict)

    ds_empty = Dataset.from_dict(data_empty_tok).with_format("torch")

    model_name = model_checkpoint.split("/")[-1]
    train_args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-framenet",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy= "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "BARTScore_f1",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=5,
        predict_with_generate=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def compute_metrics(eval_pred):
        pass

    trainer = Seq2SeqTrainer(
        model,
        train_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    if num_candidates > 1:
        preds = trainer.predict(ds_empty, do_sample=True, temperature=1.5, top_k=50, max_new_tokens=64, num_return_sequences=num_candidates)
    else:
        preds = trainer.predict(ds_empty)

    outputs = torch.from_numpy(preds.predictions)
    valid_token_range = torch.arange(len(tokenizer))
    mask = torch.isin(outputs, valid_token_range)
    # Filter outputs using mask
    filtered_outputs = outputs * mask + (~mask) * tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(filtered_outputs)
    # remove <pad> tokens
    decoded_preds = [p.replace("<pad>","") for p in decoded_preds]
    return decoded_preds

def test_empty_lus(tags, num_candidates=1):
    with open(f'SRL_data_to_augment.pickle', "rb") as f:
        data_to_augment = pickle.load(f)

    data_empty = []
    for d in data_to_augment:
        empty_id, sent_aug = d
        try:
            example = prepare_data_empty(tags, empty_id, sent_aug.lu_id, sent_aug)
            if example is not None:
                data_empty.append(example)
        except KeyError:
            pass
    if args.local_rank == 0:
        print('data loaded')

    if tags.find('zero_shot') != -1: 
        model = T5ForConditionalGeneration.from_pretrained('t5-'+tags[10:])
    else:
        model = T5ForConditionalGeneration.from_pretrained('t5-finetuned-framenet-'+tags+'-complete')
    model = model.to(device)
    model.eval()
    if args.local_rank == 0:
        print('model loaded')

    decoded_preds_total = predict_empty_batch(data_empty, model, num_candidates=num_candidates)

    # only perform generation when LU is verb and there exist masks in sentence
    decoded_preds_valid = []
    for i in range(len(decoded_preds_total)):
        j = int(math.floor(i/num_candidates))
        empty_lu = id2lu[data_empty[j]['empty_id']]
        pos = empty_lu[(empty_lu.rfind('.')+1):]
        if pos == 'v' and data_empty[j]['X'].find('extra_id') != -1:
            decoded_preds_valid.append(decoded_preds_total[i])
        else:
            decoded_preds_valid.append('')
    with open(f'y_pred_empty_{tags}_fulltext_v.pickle', 'wb') as f:
        pickle.dump(decoded_preds_valid, f)
    if args.local_rank == 0:
        print('output saved')

    return

def test_empty_lus_gpt(tags):

    with open(f'SRL_data_to_augment_fulltext_lu_wise_f1_v.pickle', "rb") as f:
        data_to_augment = pickle.load(f)

    data_empty = []
    for d in data_to_augment:
        empty_id, sent_aug = d
        try:
            data_empty.append(prepare_data_empty(tags, empty_id, sent_aug.lu_id, sent_aug))
        except KeyError:
            pass
    X = [d['X'] for d in data_empty]
    X_prompt = [d['X_prompt'] for d in data_empty]
    print("dataset loaded")

    # load checkpoint generation
    if os.path.exists(f'y_pred_empty_{tags}_fulltext_v.pickle'):
        with open(f'y_pred_empty_{tags}_fulltext_v.pickle', 'rb') as f:
            preds = pickle.load(f)
    for i in tqdm(range(len(preds),len(X_prompt))):
        p = X_prompt[i]
        # only perform generation when LU is verb and there exist masks in sentence
        empty_lu = id2lu[data_empty[i]['empty_id']]
        pos = empty_lu[(empty_lu.rfind('.')+1):]
        if pos == 'v' and X[i].find("<mask>") != -1:
            if tags == 'GPT_no_tag':
                prompt = f'- Title: Sentence completion using frame elements \n \
- Definition: You need to complete the given sentence containing one or multiple blanks (<mask>). \n \
- Positive example: \n \
- Input: Lexical Unit: bake.v. Sentence: <mask> is baked <mask> for 10 minutes and served with mushrooms. \n \
- Output: The mix, in moulds  \n\
- Reason: The answer "The mix" fills up the first blank. The answer "in moulds" fills up the second blank. \n \
- Prompt: Fill in the blanks in the sentence based on the provided lexical unit. Generate the spans that fill up the blanks ONLY. Do NOT generate the whole sentence or existing parts of the sentence. Separate the generated spans of different blanks by a comma. Generate the output of the task instance ONLY. Do NOT include existing words or phrases before or after the blank. \n \
- Task instance: \n \
Input: {p} \n \
Output: '
            elif tags == 'GPT_FE_only':
                prompt = f'- Title: Sentence completion using frame elements \n \
- Definition: You need to complete the given sentence containing one or multiple blanks (<mask>). Your answer must be of the frame element type specified in FE Type. \n \
- Positive example: \n \
- Input: Lexical Unit: bake.v. Sentence: <mask> is baked <mask> for 10 minutes and served with mushrooms. FE Type: Entity, Container. \n \
- Output: The mix, in moulds  \n\
- Reason: The answer "The mix" fills up the first blank because it is a frame element (FE) of type "Entity". The answer "in moulds" fills up the second blank because it is an FE of type "Container". \n \
- Prompt: Fill in the blanks in the sentence based on the provided lexical unit and FE type. Generate the spans that fill up the blanks ONLY. Do NOT generate the whole sentence or existing parts of the sentence. Separate the generated spans of different blanks by a comma. Generate the output of the task instance ONLY. Do NOT include existing words or phrases before or after the blank. \n \
- Task instance: \n \
Input: {p} \n \
Output: '
            elif tags == 'GPT_frame+FE':
                prompt = f'- Title: Sentence completion using frame elements \n \
- Definition: You need to complete the given sentence containing one or multiple blanks (<mask>). Your answer must be of the frame element type specified in FE Type. \n \
- Positive example: \n \
- Input: Frame: Absorb_heat. Lexical Unit: bake.v. Sentence: <mask> is baked <mask> for 10 minutes and served with mushrooms. FE Type: Entity, Container. \n \
- Output: The mix, in moulds  \n\
- Reason: The frame "Absorb_heat" is associated with frame elements "Entity" and "Container". The answer "The mix" fills up the first blank because it is a frame element (FE) of type "Entity". The answer "in moulds" fills up the second blank because it is an FE of type "Container". \n \
- Prompt: Fill in the blanks in the sentence based on the provided frame, lexical unit and FE type. Generate the spans that fill up the blanks ONLY. Do NOT generate the whole sentence or existing parts of the sentence. Separate the generated spans of different blanks by a comma. Generate the output of the task instance ONLY. Do NOT include existing words or phrases before or after the blank. \n \
- Task instance: \n \
Input: {p} \n \
Output: '
            else:
                print('invalid tags')
            preds.append(gpt_generate(prompt))
            # avoid going over OpenAI rate limit
            # time.sleep(0.5)

        else:
            preds.append('')
        with open(f'y_pred_empty_{tags}_fulltext_v.pickle', 'wb') as f:
            pickle.dump(preds, f)

    return

if __name__ == "__main__":

    ########## TRAIN ##########
    # T5
    train_model('no_tag')
    # T5 | FE
    train_model('FE_only')
    # T5 | Frame+FE
    train_model('frame+FE')

    ########## TEST TARGET EMPTY LUS ##########
    # T5
    test_empty_lus('no_tag')
    # T5 | FE
    test_empty_lus('FE_only')
    # T5 | Frame+FE
    test_empty_lus('frame+FE')

    # GPT-4
    test_empty_lus_gpt('GPT_no_tag')
    # GPT-4 | FE
    test_empty_lus_gpt('GPT_FE_only')
    # GPT-4 | Frame+FE
    test_empty_lus_gpt('GPT_frame+FE')
    