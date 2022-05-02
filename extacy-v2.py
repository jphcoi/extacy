#!/usr/bin/env python
# coding: utf-8



import random
import spacy
import textacy
import textacy.extract as extract
import extacy
from functools import partial
import math
import spacy
import tqdm
from spacy.matcher import PhraseMatcher

from spacy.tokenizer import _get_regex_pattern
import re
from operator import itemgetter

from spacymoji import Emoji

import pandas as pd
from tqdm import tqdm
import os



#################################################################@
#parameters
#################################################################@
#restrain the total number of docs and their length
N_doc_sample=1000
max_doc_size=3000 # characters
text_field='abstract'
data_file='/Users/jpcointet/Desktop/syllab/code/socsc.csv'
lang='en'

#extraction parmeters
with_NER=True
type_pattern=['NP']
NER_types=['PER','ORG']
ngrmin=1
ngrmax=3

#selection

ranking='C-value'
freq_thres=.000005
topn=1000

coherence_columns=['year']#'source','year']
case_insensitive=False


ranking='C-value'
ranking2='surprise (sample)'
name_export=str(N_doc_sample)+ranking

path0=os.getcwd()
result_path=os.path.join(path0,'indexation_results')
try:
    os.mkdir(result_path)
except:
    pass


###.1 first loading some data, we just need a list of documents
#################################################################@


df=pd.read_csv(data_file)


###.2  Preparing a sample of the original corpus and building a coprus
#################################################################@


N_doc_sample=min(N_doc_sample,len(df))
df_domain_sample=df.sample(n=N_doc_sample)
documents_sample=['.'.join(doc[:max_doc_size].split('.')[:-1]) for doc in df_domain_sample[:N_doc_sample][text_field]]
documents_sample=[doc for doc in df_domain_sample[:N_doc_sample][text_field]]





###.3 extracting key-terms
#################################################################@

#we first build a corpus using the sample of data
#parameter
nlp=extacy.build_extraction_pipe(lang,with_NER=with_NER)
corpus = textacy.Corpus(nlp, data=documents_sample)


N=len(corpus)
NW=corpus.n_tokens
#100 000 is good, 1M is overkill
print(" Total number of tokens in the corpus: ",NW)
print(" sampled over a number of documents: ",N)


#we build a collection of candidate noun phrases with at least ngrmin words
#################################################################@

sample_dictionary,sample_index = extacy.extract(corpus,lang=lang,ngrmin=ngrmin,ngrmax=ngrmax,type_pattern=type_pattern,NER_extract=with_NER,NER_types=NER_types)



# We then build a dictionnary of nested strings (useful to compute the cvalue)
#################################################################@
#ranking='C-value'
#'C-value'#'pigeon'#tfidf
if ranking=='C-value':
    #parameter
    threshold=1#if higher than 1, the pre-processing is faster, usefull if pigeon is activated
    # To accelerate things we get rid of terms which appear less than twice but this is optionnal
    sample_dictionary,sample_index=extacy.pre_filtering(sample_dictionary,sample_index,threshold=threshold)
    nested,n_dict=extacy.build_nested(sample_dictionary)# set Fast to None for an exhaustive search, fast=2 to go slightly faster, fast=5 to remove
else:
    n_dict=extacy.build_n_dict(sample_dictionary)
    nested={}


# In[724]:


import os
# Selecting 2 * topn based on ranking
#################################################################@
word_list_df,sample_dictionary_main=extacy.selecting(corpus,nested,sample_dictionary,sample_index,n_dict,freq_thres=freq_thres,topn=int(2*topn),ranking=ranking)
word_list_df.to_csv(os.path.join(result_path,'extracted_list_sample'+'_'+ranking+'_'+str(freq_thres)+'_'+str(topn)+name_export+'.csv'),index=False)


# Indexation of the sample in progress
#################################################################@
nlp,matcher=extacy.build_indexation_pipe(lang)
matcher, minimal_query=extacy.feed_matcher(nlp,matcher,sample_dictionary,sample_dictionary_main)
count,count_doc,forms_dict=extacy.index(documents_sample,sample_dictionary_main,nlp,matcher,type_export='minimal',name=name_export,nested=nested)#nested required to not do double dipping when indexing nested strings






#basic count & surprise statistics
#################################################################@
count_doc_unique={}
for w in count_doc:
    count_doc_unique[w]=len(set(count_doc[w]))



mappings=[]
for col in coherence_columns:
    sources=list(df_domain_sample[col].value_counts().keys())
    mapping={}
    for i,source in enumerate(df_domain_sample[col]):
        mapping[i]=sources.index(source)
    mappings.append(mapping)


surprise_dict=extacy.compute_surprise(documents_sample,count_doc)
word_list_df['surprise (sample)']=word_list_df['lemma'].apply(surprise_dict.get)
print ('coherence_columns',coherence_columns)
for mapping,col_name in zip(mappings,coherence_columns):
    surprise_dict=extacy.compute_surprise(documents_sample,count_doc,mapping=mapping)
    word_list_df['surprise '+col_name+' (sample)']=word_list_df['lemma'].apply(surprise_dict.get)
    disting_x={}
    for word in tqdm(count_doc,total=len(count_doc)):
        disting_x[word]=len(extacy.dist_it(count_doc[word],mapping))
    word_list_df['indexed distinct '+col_name+' (sample)']=word_list_df['lemma'].map(disting_x)


word_list_df['indexed occurrences (sample)']=word_list_df['lemma'].map(count)
word_list_df['indexed distinct documents (sample)']=word_list_df['lemma'].map(count_doc_unique)


ratio_count={}
for w,f in zip(word_list_df['lemma'],word_list_df['occurrences']):
    #print(w,count.get(w,0),f)
    ratio_count[w]=count.get(w,0)/f
print("\nover-represented:\t",sorted(ratio_count.items(),key=itemgetter(1),reverse=True)[:20])
print("\nunder-represented:\t",sorted(ratio_count.items(),key=itemgetter(1),reverse=False)[:20])
print("\nfrequent:\t",sorted(count.items(),key=itemgetter(1),reverse=True)[:20])
print("\ncandidates:",list(map(lambda x:x[0],sorted(ratio_count.items(),key=itemgetter(1),reverse=True)[:20]))+list(map(lambda x:x[0],sorted(ratio_count.items(),key=itemgetter(1),reverse=False)[:20])))
stopwords = [x[0] for x in  ratio_count.items() if (x[1]<0.1) | (x[1]>4)]

print('\nunder/over:',stopwords)


#Final selection of key words
#################################################################@
documents=df[text_field]
#parameters:
word_list_df.drop(word_list_df[word_list_df.lemma.isin(stopwords)].index, inplace=True)
print(len(word_list_df))
word_list_df=extacy.homogenize_wl(word_list_df)
print("afterh",len(word_list_df))
word_list_df.sort_values(by=[ranking2], inplace=True,ascending=False)

word_list_df=word_list_df[:topn]
print("finallen",len(word_list_df))


#########################################################################@
###5. indexing the full text content
#termlist=os.path.join(result_path,'extracted_list_nostop'+'_'+ranking+'_'+str(freq_thres)+'_'+str(topn)+'.csv')
#require columns lemma, main_word and words (dict style)
#word_list_df=pd.read_csv(termlist)
#########################################################################@
len(word_list_df)


sample_dictionary_main=dict(zip(word_list_df['lemma'],word_list_df['main_word']))
sample_dictionary=dict(zip(word_list_df['lemma'],word_list_df['words']))
for x in sample_dictionary:
    sample_dictionary[x]=eval(str(sample_dictionary[x]))

nlp,matcher=extacy.build_indexation_pipe(lang,case_insensitive=case_insensitive)
matcher, minimal_query=extacy.feed_matcher(nlp,matcher,sample_dictionary,sample_dictionary_main)
count,count_doc,dict_forms=extacy.index(documents,sample_dictionary_main,nlp,matcher,type_export='simple')


count_doc_unique={}
for w in count_doc:
    count_doc_unique[w]=len(set(count_doc[w]))

#word_list_df=pd.read_csv('extracted_list_nostop'+'_'+ranking+'_'+str(freq_thres)+'_'+str(topn)+'_'+domain+'.csv')
sample_dictionary_main=dict(zip(word_list_df['lemma'],word_list_df['main_word']))
sample_dictionary=dict(zip(word_list_df['lemma'],word_list_df['words']))

for x in sample_dictionary:
    sample_dictionary[x]=eval(str(sample_dictionary[x]))



word_list_df['indexed occurrences']=word_list_df['lemma'].map(count)
word_list_df['indexed documents']=word_list_df['lemma'].map(count_doc_unique)
word_list_df['minimal_query']=word_list_df['lemma'].map(minimal_query)
word_list_df['full_index_forms']=word_list_df['lemma'].map(dict_forms)




word_list_df.to_csv(os.path.join(result_path,'extracted_list_full'+'_'+ranking+'_'+str(freq_thres)+'_'+str(topn)+'.csv'),index=False)
