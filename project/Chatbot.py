#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 00:51:26 2019

@author: sadasivam
"""
import pytextrank
import json
import nltk
import warnings
warnings.filterwarnings("ignore")

# nltk.download() # for downloading packages


import random
import string # to process standard python strings



f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words


sent_tokens[:2]


word_tokens[:5]


lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]



# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
final_text = ''
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                print("ROBO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
                final_text = final_text + '\n' + user_response + '\n' + response(user_response) + '\n' 
                
    else:
        flag=False
        print("ROBO: Bye! take care..")    
        
text = open("Conversation.txt", "w")
text.write(final_text)
text.close
sample_text = final_text
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
