#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:19:13 2019

@author: sadasivam
"""

from sklearn.feature_extraction.text import TfidfVectorizer

documents = []

file1 = open("NLPTask1.txt","r").read()
file2 = open("NLPTask2.txt","r").read()
file3 = open("NLPTask3.txt","r").read()


corpus = [
    file1,file2,file3
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
Y = vectorizer.fit_transform(corpus)
yArr = (Y * Y.T).A
cos1 = sum(yArr[0])
cos2 = sum(yArr[1])
cos3 = sum(yArr[2])
print(X.shape)
print(cos1,cos2,cos3)
