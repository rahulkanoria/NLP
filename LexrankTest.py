#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 23:17:31 2019

@author: sadasivam
"""

from lexrank import STOPWORDS, LexRank
from path import Path

documents = []
documents_dir = Path('bbc/politics')

for file_path in documents_dir.files('*.txt'):
    with file_path.open(mode='rt', encoding='utf-8') as fp:
        documents.append(fp.readlines())

lxr = LexRank(documents, stopwords=STOPWORDS['en'])

sentences = [
    'One of David Cameron\'s closest friends and Conservative allies, '
    'George Osborne rose rapidly after becoming MP for Tatton in 2001.',

    'Michael Howard promoted him from shadow chief secretary to the '
    'Treasury to shadow chancellor in May 2005, at the age of 34.',

    'Mr Osborne took a key role in the election campaign and has been at '
    'the forefront of the debate on how to deal with the recession and '
    'the UK\'s spending deficit.',

    'Even before Mr Cameron became leader the two were being likened to '
    'Labour\'s Blair/Brown duo. The two have emulated them by becoming '
    'prime minister and chancellor, but will want to avoid the spats.',

    'Before entering Parliament, he was a special adviser in the '
    'agriculture department when the Tories were in government and later '
    'served as political secretary to William Hague.',

    'The BBC understands that as chancellor, Mr Osborne, along with the '
    'Treasury will retain responsibility for overseeing banks and '
    'financial regulation.',

    'Mr Osborne said the coalition government was planning to change the '
    'tax system \"to make it fairer for people on low and middle '
    'incomes\", and undertake \"long-term structural reform\" of the '
    'banking sector, education and the welfare state.',
]

# get summary with classical LexRank algorithm
summary = lxr.get_summary(sentences, summary_size=2, threshold=.1)
print(summary)

# ['Mr Osborne said the coalition government was planning to change the tax '
#  'system "to make it fairer for people on low and middle incomes", and '
#  'undertake "long-term structural reform" of the banking sector, education and '
#  'the welfare state.',
#  'The BBC understands that as chancellor, Mr Osborne, along with the Treasury '
#  'will retain responsibility for overseeing banks and financial regulation.']


# get summary with continuous LexRank
summary_cont = lxr.get_summary(sentences, threshold=None)
print(summary_cont)

# ['The BBC understands that as chancellor, Mr Osborne, along with the Treasury '
#  'will retain responsibility for overseeing banks and financial regulation.']

# get LexRank scores for sentences
# 'fast_power_method' speeds up the calculation, but requires more RAM
scores_cont = lxr.rank_sentences(
    sentences,
    threshold=None,
    fast_power_method=False,
)
print(scores_cont)

#  [1.0896493024505858,
#  0.9010711968859021,
#  1.1139166497016315,
#  0.8279523250808547,
#  0.8112028559566362,
#  1.185228912485382,
#  1.0709787574388283]