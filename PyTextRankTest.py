#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:09:24 2019

@author: sadasivam
"""

# Pytextrank
import pytextrank
import json
from lexrank import STOPWORDS, LexRank
from path import Path

sample_text = ''
documents = []
documents_dir = Path('bbc')

for file_path in documents_dir.files('*.txt'):
    file = open(file_path,'r')
    text = file.read()
    sample_text+=text

# Sample text
#sample_text = 'Mr Osborne said the coalition government was planning to change the tax system "to make it fairer for people on low and middle incomes", and undertake "long-term structural reform" of the banking sector, education and the welfare state.' \
#'The BBC understands that as chancellor, Mr Osborne, along with the Treasury will retain responsibility for overseeing banks and financial regulation.'


# Create dictionary to feed into json file

file_dic = {"id" : 0,"text" : sample_text}
file_dic = json.dumps(file_dic)
loaded_file_dic = json.loads(file_dic)

# Create test.json and feed file_dic into it.
with open('test.json', 'w') as outfile:
        json.dump(loaded_file_dic, outfile)

path_stage0 = "test.json"
path_stage1 = "o1.json"

# Extract keyword using pytextrank
with open(path_stage1, 'w') as f:
    for graf in pytextrank.parse_doc(pytextrank.json_iter(path_stage0)):
        f.write("%s\n" % pytextrank.pretty_print(graf._asdict()))
        #print(pytextrank.pretty_print(graf._asdict()))

path_stage1 = "o1.json"
path_stage2 = "o2.json"

graph, ranks = pytextrank.text_rank(path_stage1)
pytextrank.render_ranks(graph, ranks)

with open(path_stage2, 'w') as f:
    for rl in pytextrank.normalize_key_phrases(path_stage1, ranks):
        f.write("%s\n" % pytextrank.pretty_print(rl._asdict()))
        #print(pytextrank.pretty_print(rl))

path_stage1 = "o1.json"
path_stage2 = "o2.json"
path_stage3 = "o3.json"

kernel = pytextrank.rank_kernel(path_stage2)

with open(path_stage3, 'w') as f:
    for s in pytextrank.top_sentences(kernel, path_stage1):
        f.write(pytextrank.pretty_print(s._asdict()))
        f.write("\n")
        # to view output in this notebook
        print(pytextrank.pretty_print(s._asdict()))

path_stage2 = "o2.json"
path_stage3 = "o3.json"

phrases = ", ".join(set([p for p in pytextrank.limit_keyphrases(path_stage2, phrase_limit=20)]))
sent_iter = sorted(pytextrank.limit_sentences(path_stage3, word_limit=500), key=lambda x: x[1])
s = []

for sent_text, idx in sent_iter:
    s.append(pytextrank.make_sentence(sent_text))

graf_text = " ".join(s)
print("**excerpts:** %s\n\n**keywords:** %s" % (graf_text, phrases,))