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


ns = {'fn' : 'http://framenet.icsi.berkeley.edu'}

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
with open("lu2sents.pickle", "rb") as f:
    lu2sents = pickle.load(f)
with open("id2lu.pickle", "rb") as f:
    id2lu = pickle.load(f)

TEST_FILES = [
        "ANC__110CYL067.xml",
        "ANC__110CYL069.xml",
        "ANC__112C-L013.xml",
        "ANC__IntroHongKong.xml",
        "ANC__StephanopoulosCrimes.xml",
        "ANC__WhereToHongKong.xml",
        "KBEval__atm.xml",
        "KBEval__Brandeis.xml",
        "KBEval__cycorp.xml",
        "KBEval__parc.xml",
        "KBEval__Stanford.xml",
        "KBEval__utd-icsi.xml",
        "LUCorpus-v0.3__20000410_nyt-NEW.xml",
        "LUCorpus-v0.3__AFGP-2002-602187-Trans.xml",
        "LUCorpus-v0.3__enron-thread-159550.xml",
        "LUCorpus-v0.3__IZ-060316-01-Trans-1.xml",
        "LUCorpus-v0.3__SNO-525.xml",
        "LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml",
        "Miscellaneous__Hound-Ch14.xml",
        "Miscellaneous__SadatAssassination.xml",
        "NTI__NorthKorea_Introduction.xml",
        "NTI__Syria_NuclearOverview.xml",
        "PropBank__AetnaLifeAndCasualty.xml",
        ]

DEV_FILES = [
        "ANC__110CYL072.xml",
        "KBEval__MIT.xml",
        "LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
        "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml",
        "Miscellaneous__Hijack.xml",
        "NTI__NorthKorea_NuclearOverview.xml",
        "NTI__WMDNews_062606.xml",
        "PropBank__TicketSplitting.xml",
        ]


def preprocess_data_split(filepath, filelist):
    # format: list of SentenceFEs
    data = []

    for filename in tqdm(filelist):
        path = filepath + filename
        with codecs.open(path, 'rb', 'utf-8') as xml_file:
            tree = et.parse(xml_file)
        root = tree.getroot()
        for sentence in root.iter('{http://framenet.icsi.berkeley.edu}sentence'):
            sent_id = sentence.attrib['ID']
            for annotation in sentence.iter('{http://framenet.icsi.berkeley.edu}annotationSet'):
                if annotation.attrib['ID'] == "2019791":
                    # Hack to skip an erroneous annotation of Cathedral as raise.v with frame "Growing_food".
                    continue
                if 'luID' in annotation.attrib:
                    # Ignore unannotated instances
                    if annotation.attrib["status"] == "UNANN":
                        continue
                    lu_id = annotation.attrib['luID']
                    # find sentence in lu2sents
                    for sentanno in lu2sents[lu_id]:
                        if sentanno.id == sent_id:
                            data.append(sentanno)
                            break
    return data

def preprocess_train_data():
    with open('SRL_data_val_sentanno.pickle', 'rb') as f:
        data_val = pickle.load(f)
    with open('SRL_data_test_sentanno.pickle', 'rb') as f:
        data_test = pickle.load(f)
    data_val_sent_ids = set([sentanno.id for sentanno in data_val])
    data_test_sent_ids = set([sentanno.id for sentanno in data_test])
    data_train = []
    for lu_id in id2lu:
        # LU is not empty
        if lu_id in lu2sents:
            for sentanno in lu2sents[lu_id]:
                if sentanno.id not in data_val_sent_ids and sentanno.id not in data_test_sent_ids:
                    data_train.append(sentanno)
    return data_train

def preprocess_train_data_fulltext(filename):
    data = []
    #print(luIndex_file)
    with codecs.open(filename, 'rb', 'utf-8') as xml_file:
        tree = et.parse(xml_file)
    root = tree.getroot()

    for corpus in tqdm(root.findall('fn:corpus', ns)):
        corpus_name = corpus.attrib['name']
        for document in corpus.findall('fn:document', ns):
            document_name = document.attrib['name']
            filename = f'{corpus_name}__{document_name}.xml'
            if filename in DEV_FILES or filename in TEST_FILES:
                continue
            with codecs.open(f'fndata-1.7/fulltext/{filename}', 'rb', 'utf-8') as file:
                print(f'fndata-1.7/fulltext/{filename}')
                tree_fulltext = et.parse(file)
            root_fulltext = tree_fulltext.getroot()
            for sentence in root_fulltext.iter('{http://framenet.icsi.berkeley.edu}sentence'):
                sent_id = sentence.attrib['ID']
                for annotation in sentence.iter('{http://framenet.icsi.berkeley.edu}annotationSet'):
                    if annotation.attrib['ID'] == "2019791":
                        # Hack to skip an erroneous annotation of Cathedral as raise.v with frame "Growing_food".
                        continue
                    if 'luID' in annotation.attrib:
                        # Ignore unannotated instances
                        if annotation.attrib["status"] == "UNANN":
                            continue
                        lu_id = annotation.attrib['luID']
                        # find sentence in lu2sents
                        for sentanno in lu2sents[lu_id]:
                            if sentanno.id == sent_id:
                                data.append(sentanno)
                                break
    return data

if __name__ == "__main__":

    data_val = preprocess_data_split('fndata-1.7/fulltext/', DEV_FILES)
    with open('SRL_data_val_sentanno.pickle', 'wb') as f:
        pickle.dump(data_val, f)
    data_test = preprocess_data_split('fndata-1.7/fulltext/', TEST_FILES)
    with open('SRL_data_test_sentanno.pickle', 'wb') as f:
        pickle.dump(data_test, f)

    data_train = preprocess_train_data()
    with open('SRL_data_train_sentanno.pickle', 'wb') as f:
        pickle.dump(data_train, f)
    data_train_fulltext = preprocess_train_data_fulltext('fndata-1.7/fulltextIndex.xml')
    with open('SRL_data_train_fulltext_sentanno.pickle', 'wb') as f:
        pickle.dump(data_train_fulltext, f)

    # print(len(data_train))
    # print(len(data_val))
    # print(len(data_test))






