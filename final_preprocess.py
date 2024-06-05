import pickle
import sys
from __future__ import division

'''
Reads XML files containing FrameNet 1.$VERSION annotations, and converts them to a CoNLL 2009-like format.
'''
import codecs
import os.path

import importlib
importlib.reload(sys)

from tqdm import tqdm
import pandas as pd
import numpy as np
import xml.etree.ElementTree as et
from optparse import OptionParser

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


ns = {'fn' : 'http://framenet.icsi.berkeley.edu'}
relevantfelayers = ["FE", "PT"]

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

    def process_target_inds(self):
        # merge multi-word spans
        stinds = sorted(self.target_stinds)
        eninds = sorted(self.target_eninds)
        stinds_merged = []
        eninds_merged = []
        start = stinds[0]
        end = eninds[0]
        for i in range(1, len(stinds)):
            # merge current word with previous span
            if stinds[i] - end <= 2:
                end = eninds[i]
            # current word is start of new span
            else:
                # record previous span
                stinds_merged.append(start)
                eninds_merged.append(end)
                # restart
                start = stinds[i]
                end = eninds[i]
        # record current span
        stinds_merged.append(start)
        eninds_merged.append(end)

        self.target_stinds = sorted(stinds_merged)
        self.target_eninds = sorted(eninds_merged)

def process_xml_labels(label, layertype):
    try:
        st = int(label.attrib["start"])
        en = int(label.attrib["end"])
    except KeyError:
        if layertype != 'FE':
          print("\t\tIssue: start and/or end labels missing in " + layertype + "\n")
        return
    return (st, en)

def anno_fes_in_sent(sent, lu_id, core_fes):
    sent_id = sent.attrib['ID']
    for t in sent.findall('fn:text', ns):  # not a real loop
        senttext = t.text
        # print(t.text)
    sentanno = SentenceFEs(lu_id, sent_id, senttext, core_fes)
    phrase_types = []
    for anno in sent.findall('fn:annotationSet', ns):
        if anno.attrib['status'] == 'UNANN':
            continue
        # find target first
        for layer in anno.findall('fn:layer', ns):
            layertype = layer.attrib["name"]
            if layertype == "Target":
                for label in layer.findall('fn:label', ns):  # can be a real loop
                    startend = process_xml_labels(label, layertype)
                    if startend is None:
                        print('invalid Target indices')
                        continue
                    sentanno.add_target(startend[0], startend[1])
                # target not found
                if len(sentanno.target_stinds) == 0:
                    break
                # merge multi-word target spans
                sentanno.process_target_inds()

                # find fes and phrase types
                for layer in anno.findall('fn:layer', ns):
                    layertype = layer.attrib["name"]
                    if layertype not in relevantfelayers:
                        continue
                    elif layer.attrib["name"] == "FE" and layer.attrib["rank"] == "1":
                        for label in layer.findall('fn:label', ns):
                            startend = process_xml_labels(label, layertype)
                            if startend is None:
                                if "itype" in label.attrib:
                                    # print("\t\tIssue: itype = " + label.attrib["itype"] + "\n")
                                    continue
                                else:
                                    break
                            # overlap with target
                            overlap_target = False
                            for i in range(len(sentanno.target_stinds)):
                                if sentanno.target_stinds[i] <= startend[0] and startend[0] <= sentanno.target_eninds[i]:
                                    overlap_target = True
                                    break
                            if overlap_target:
                                continue
                            sentanno.add_fe(label.attrib['name'], label.attrib['feID'], startend[0], startend[1])

                    elif layer.attrib["name"] == "PT" and layer.attrib["rank"] == "1":
                        for label in layer.findall('fn:label', ns):
                            startend = process_xml_labels(label, layertype)
                            if startend is None:
                                if "itype" in label.attrib:
                                    # print("\t\tIssue: itype = " + label.attrib["itype"] + "\n")
                                    continue
                                else:
                                    break
                            # print(label.attrib['name'])
                            sentanno.add_fe_phrasetype(startend[0], label.attrib['name'])

    return sentanno

def create_mapping(luIndex_file):
    #print(luIndex_file)
    with codecs.open(luIndex_file, 'rb', 'utf-8') as xml_file:
        tree = et.parse(xml_file)
    root = tree.getroot()
    # frame id to all corresponding LU ids
    frame2lus = {}
    # LU id to their frame id
    lu2frame = {}
    # non-empty LU ids to their annotated sentences
    lu2sents = {}
    # LU id to LU name
    id2lu = {}
    # frame id to frame name
    id2frame = {}

    for lu in tqdm(root.findall('fn:lu', ns)):
        # print(count)
        lu_id = lu.attrib['ID']
        id2lu[lu_id] = lu.attrib['name']
        frame_id = lu.attrib['frameID']
        id2frame[frame_id] = lu.attrib['frameName']
        if frame_id in frame2lus:
            # append LU id and whether it's nonempty
            frame2lus[frame_id].append(lu_id)
        else:
            frame2lus[frame_id] = [lu_id]
        lu2frame[lu_id] = frame_id
        # print(lu.attrib['name'])

        if lu.attrib['hasAnnotation']:
            # goto corresponding lu file
            with codecs.open('fndata-1.7/lu/lu'+lu_id+'.xml', 'rb', 'utf-8') as file:
                tree_lu = et.parse(file)
            root_lu = tree_lu.getroot()
            
            # core FEs under current frame
            core_fes = set()
            for frame in root_lu.iter('{http://framenet.icsi.berkeley.edu}frame'):  # not a real loop
                for fe_type in frame.findall('fn:FE', ns):
                    if fe_type.attrib['type'] == "Core":
                        core_fes.add(fe_type.attrib['name'])

            for sent in root_lu.iter('{http://framenet.icsi.berkeley.edu}sentence'):
                sentanno = anno_fes_in_sent(sent, lu_id, core_fes)
                if lu_id not in lu2sents:
                    lu2sents[lu_id] = [sentanno]
                else:
                    lu2sents[lu_id].append(sentanno)
    core_fes = set()
    # all core FE ids
    for lu_id, sentanno in lu2sents.items():
        for fe in sentanno.fes:
            if fe['is_core']:
                core_fes.add(fe['id'])
    core_fes = list(core_fes)

    return frame2lus, lu2frame, lu2sents, id2lu, id2frame, core_fes

def anno_fes_in_sent_fulltext(sent, core_fes):
    sent_id = sent.attrib['ID']
    for t in sent.findall('fn:text', ns):  # not a real loop
        senttext = t.text
    sentanno_list = []
    frame_list = []
    lu_name_list = []
    for anno in sent.findall('fn:annotationSet', ns):
        if anno.attrib['ID'] == "2019791":
            # Hack to skip an erroneous annotation of Cathedral as raise.v with frame "Growing_food".
            continue
        if 'luID' in anno.attrib:
            # Ignore unannotated instances
            if anno.attrib["status"] == "UNANN":
                # print('unannotated example')
                continue
            lu_id = anno.attrib['luID']
            frame_id = anno.attrib['frameID']
            lu_name = anno.attrib['luName']
            frame_list.append(frame_id)
            lu_name_list.append(lu_name)
            sentanno = SentenceFEs(lu_id, sent_id, senttext, core_fes)
            phrase_types = []

            # find target first
            for layer in anno.findall('fn:layer', ns):
                layertype = layer.attrib["name"]
                if layertype == "Target" and layer.attrib['rank'] == '1':
                    for label in layer.findall('fn:label', ns):  # can be a real loop
                        startend = process_xml_labels(label, layertype)
                        if startend is None:
                            break
                        sentanno.add_target(startend[0], startend[1])
                    # target not found
                    if len(sentanno.target_stinds) == 0:
                        print('target not found')
                        continue
            sentanno.process_target_inds()
            # find fes
            for layer in anno.findall('fn:layer', ns):
                layertype = layer.attrib["name"]
                if layertype not in relevantfelayers:
                    continue
                elif layer.attrib["name"] == "FE" and layer.attrib["rank"] == "1":
                    for label in layer.findall('fn:label', ns):
                        startend = process_xml_labels(label, layertype)
                        if startend is None:
                            if "itype" in label.attrib:
                                # print("\t\tIssue: itype = " + label.attrib["itype"] + "\n")
                                continue
                            else:
                                break
                        # overlap with target
                        overlap_target = False
                        for i in range(len(sentanno.target_stinds)):
                            if sentanno.target_stinds[i] <= startend[0] and startend[0] <= sentanno.target_eninds[i]:
                                overlap_target = True
                                break
                        # sentanno.add_fe(label.attrib["name"], label.attrib["feID"], startend[0], startend[1])
                        # print(label.attrib['name'])
                        if overlap_target:
                            continue
                        # sentanno.add_fe(label.attrib["name"], label.attrib["feID"], startend[0], startend[1])
                        # print(label.attrib['name'])
                        sentanno.add_fe(label.attrib['name'], label.attrib['feID'], startend[0], startend[1])

                elif layer.attrib["name"] == "PT" and layer.attrib["rank"] == "1":
                    for label in layer.findall('fn:label', ns):
                        startend = process_xml_labels(label, layertype)
                        if startend is None:
                            if "itype" in label.attrib:
                                # print("\t\tIssue: itype = " + label.attrib["itype"] + "\n")
                                continue
                            else:
                                break
                        # print(label.attrib['name'])
                        sentanno.add_fe_phrasetype(startend[0], label.attrib['name'])
            sentanno_list.append(sentanno)

    return sentanno_list, frame_list, lu_name_list

def create_mapping_fulltext(filename, core_fes, frame2lus, lu2frame, lu2sents, id2lu, fe2frame):
    core_fes = set([id2fe[fe_id] for fe_id in core_fes])
    with codecs.open(filename, 'rb', 'utf-8') as xml_file:
        tree = et.parse(xml_file)
    root = tree.getroot()

    for corpus in tqdm(root.findall('fn:corpus', ns)):
        corpus_name = corpus.attrib['name']
        for document in corpus.findall('fn:document', ns):
            document_name = document.attrib['name']
            with codecs.open(f'fndata-1.7/fulltext/{corpus_name}__{document_name}.xml', 'rb', 'utf-8') as file:
                # print(f'fndata-1.7/fulltext/{corpus_name}__{document_name}.xml')
                tree_fulltext = et.parse(file)
            root_fulltext = tree_fulltext.getroot()
            for sentence in root_fulltext.findall('fn:sentence', ns):
                sentanno_list, frame_list, lu_name_list = anno_fes_in_sent_fulltext(sentence, core_fes)
                for i, sentanno in enumerate(sentanno_list):
                    lu_id = sentanno.lu_id
                    frame_id = frame_list[i]
                    lu_name = lu_name_list[i]
                    sent_id = sentanno.id
                    # update unseen LU
                    if lu_id not in id2lu:
                        frame2lus[frame_id].append(lu_id)
                        lu2frame[lu_id] = frame_id
                        lu2sents[lu_id] = []
                        id2lu[lu_id] = lu_name
                    if lu_id not in lu2sents:
                        lu2sents[lu_id] = []
                    # update unseen sentanno
                    seen = False
                    for anno in lu2sents[lu_id]:
                        if sent_id == anno.id:
                            seen = True
                            break
                    if not seen:
                        lu2sents[lu_id].append(sentanno)

    return frame2lus, lu2frame, lu2sents, id2lu

def create_frame_fe_mapping(filename):
    #print(luIndex_file)
    with codecs.open(filename, 'rb', 'utf-8') as xml_file:
        tree = et.parse(xml_file)
    root = tree.getroot()
    frame2fes = {}
    fe2frame = {}
    id2fe = {}

    count = 0
    for frame in tqdm(root.findall('fn:frame', ns)):
        # print(count)
        frame_id = frame.attrib['ID']
        frame_name = frame.attrib['name']

        # goto corresponding frame file
        with codecs.open(f'fndata-1.7/frame/{frame_name}.xml', 'rb', 'utf-8') as file:
            tree_frame = et.parse(file)
        root_frame = tree_frame.getroot()
        for frame in root_frame.iter('{http://framenet.icsi.berkeley.edu}frame'):  # not a real loop
            for fe in frame.findall('fn:FE', ns):
                fe_id = fe.attrib['ID']
                if frame_id in frame2fes:
                    frame2fes[frame_id].append(fe_id)
                else:
                    frame2fes[frame_id] = [fe_id]
                fe2frame[fe_id] = frame_id
                id2fe[fe_id] = fe.attrib['name']

    return frame2fes, fe2frame, id2fe

def parse_fr_relation(file_path):
    with codecs.open(file_path, 'rb', 'utf-8') as xml_file:
        tree = et.parse(xml_file)
    root = tree.getroot()
    
    fe_relations = []
    for frame_relation_type in root.findall('fn:frameRelationType', ns):
        for frame_relation in frame_relation_type.findall('fn:frameRelation', ns):
            for fe_relation in frame_relation.findall('fn:FERelation', ns):
                sub_id = fe_relation.get('subID')
                sup_id = fe_relation.get('supID')
                sub_fe_name = fe_relation.get('subFEName')
                super_fe_name = fe_relation.get('superFEName')
                fe_relations.append((sub_id, sup_id, sub_fe_name, super_fe_name))
    
    return fe_relations

def build_parent_child_map(fe_relations):
    parent_map = {}
    child_map = {}
    
    for sub_id, sup_id, sub_fe_name, super_fe_name in fe_relations:
        if sup_id not in parent_map:
            parent_map[sup_id] = []
        parent_map[sup_id].append(sub_id)
        
        if sub_id not in child_map:
            child_map[sub_id] = []
        child_map[sub_id].append(sup_id)
    
    return parent_map, child_map

def find_ancestors(frame_id, child_map):
    ancestors = set()
    stack = [frame_id]
    
    while stack:
        current = stack.pop()
        if current in child_map:
            for parent in child_map[current]:
                if parent not in ancestors:
                    ancestors.add(parent)
                    stack.append(parent)
    
    return ancestors

def get_candidate_FEs(filename, id2fe, lu2sents, id2lu):
    fe_relations = parse_fr_relation(filename)
    
    parent_map, child_map = build_parent_child_map(fe_relations)
    
    # Find all frame IDs with names "Agent" or "Self_mover"
    agent_ids = set()
    self_mover_ids = set()
    
    for sub_id, sup_id, sub_fe_name, super_fe_name in fe_relations:
        if super_fe_name == "Agent":
            agent_ids.add(sup_id)
        elif super_fe_name == "Self_mover":
            self_mover_ids.add(sup_id)
    
    # Find all descendants of "Agent" or "Self_mover"
    descendants = set()
    
    for agent_id in agent_ids:
        descendants.update(find_ancestors(agent_id, child_map))
    
    for self_mover_id in self_mover_ids:
        descendants.update(find_ancestors(self_mover_id, child_map))
    
    # Collect all FE IDs that do not have ancestors "Agent" or "Self_mover"
    all_ids = set(id2fe.keys())
    non_descendants = all_ids - descendants
    
    candidate_fes = set()
    for lu_id, sents in lu2sents.items():
        for sentanno in sents:
            lu_name = id2lu[lu_id]
            pos = lu_name[(lu_name.rfind('.')+1):]
            for fe in sentanno.fes:
                try:
                    # candidate FE
                    if pos == 'v' and fe['is_core'] and (fe['phrase_type'] == "PP" or fe['id'] in non_descendants):
                        candidate_fes.add(fe['id'])
                except:
                    pass
    candidate_fes = list(candidate_fes)
    return candidate_fes


if __name__ == "__main__":
    # create mapping for lexicographic data
    frame2lus, lu2frame, lu2sents, id2lu, id2frame, core_fes = create_mapping('fndata-1.7/luIndex.xml')
    # create mapping for fulltext data that's not included in lexicographic data
    frame2lus, lu2frame, lu2sents, id2lu, id2frame = create_mapping_fulltext('fndata-1.7/fulltextIndex.xml', core_fes, frame2lus, lu2frame, lu2sents, id2lu, fe2frame)
    # create mapping for frame and FEs
    frame2fes, fe2frame, id2fe = create_frame_fe_mapping('fndata-1.7/frameIndex.xml')
    candidate_fes = get_candidate_FEs('fndata-1.7/frRelation.xml', filename, id2fe, lu2sents, id2lu)
    fe_names = list(set(id2fe.values()))

    mappings = {"frame2lus":frame2lus, "lu2frame":lu2frame, "lu2sents":lu2sents,\
                "id2lu":id2lu, "id2frame":id2frame, "core_fes":core_fes,\
                 "frame2fes":frame2fes, "fe2frame":fe2frame, "id2fe":id2fe}
    for key, val in mappings.items():
        with open(key+'.pickle', 'wb') as f:
            pickle.dump(val, f)


