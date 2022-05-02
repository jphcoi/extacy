import spacy
from spacy.tokenizer import _get_regex_pattern
import re
from spacymoji import Emoji
import textacy
import math
from operator import itemgetter
import pandas as pd
from spacy.matcher import PhraseMatcher
from tqdm import tqdm
from scipy.special import rel_entr
import numpy as np
from scipy.spatial.distance import euclidean
import random
from functools import partial
_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64

from itertools import chain




def build_extraction_pipe(lang,with_NER=False):
	if lang=='en':
		model = "en_core_web_sm"
	elif lang=='fr':
		model= "fr_core_news_sm"
	if with_NER:
		nlp=spacy.load(model, disable=("parser"))
	else:
		nlp=spacy.load(model, disable=("parser","ner"))
	nlp.add_pipe("emoji", first=True)
	#protect hashtags and mentions
	# get default pattern for tokens that don't get split
	re_token_match = _get_regex_pattern(nlp.Defaults.token_match)
	# add your patterns (here: hashtags and in-word hyphens)
	#re_token_match = f"({re_token_match}|@\w+|#\w+|\w+-\w+)"
	re_token_match = f"({re_token_match}|@[A-Za-z]+|#[A-Za-z]+|[A-Za-z]+-[A-Za-z]+)"
	# overwrite token_match function of the tokenizer
	nlp.tokenizer.token_match = re.compile(re_token_match).match



	# Add attribute ruler with exception for hahstags
	ruler = nlp.get_pipe("attribute_ruler")
	#patterns = [[{"TEXT": {"REGEX":r"[\#|\@]\d*[A-Za-z_]+"}}]]
	patterns = [[{"TEXT": {"REGEX":r"[\#|\@][A-Za-z]+"}}]]
	#pattern = [{"TEXT": {"REGEX": "deff?in[ia]tely"}}]
	# The attributes to assign to the matched token
	attrs = {"POS": "NOUN",'TAG':"HTG"}
	# Add rules to the attribute ruler
	ruler.add(patterns=patterns, attrs=attrs, index=0)  # "The" in "The Who"
	#ruler.add(patterns=patterns, attrs=attrs, index=1)  # "Who" in "The Who"
	return nlp

def build_indexation_pipe(lang,case_insensitive=True):
	if lang=='en':
		model = "en_core_web_sm"
	elif lang=='fr':
		model= "fr_core_news_sm"
		#model='fr_core_news_lg'
	nlp=spacy.load(model,disable=('parser','ner','tok2vec','tagger','morphologizer','lemmatizer'))
	nlp.add_pipe('sentencizer')
	nlp.add_pipe("emoji", first=True)


	#protect hashtags and mentions
	# get default pattern for tokens that don't get split
	re_token_match = _get_regex_pattern(nlp.Defaults.token_match)
	# add your patterns (here: hashtags and in-word hyphens)
	#re_token_match = f"({re_token_match}|@\w+|#\w+|\w+-\w+)"
	re_token_match = f"({re_token_match}|@[A-Za-z]+|#[A-Za-z]+|[A-Za-z]+-[A-Za-z]+)"
	# overwrite token_match function of the tokenizer
	nlp.tokenizer.token_match = re.compile(re_token_match).match



# Add attribute ruler with exception for hahstags
#ruler = nlp.get_pipe("attribute_ruler")
#patterns = [[{"TEXT": {"REGEX":r"[\#|\@]\d*[A-Za-z_]+"}}]]
#pattern = [{"TEXT": {"REGEX": "deff?in[ia]tely"}}]
# The attributes to assign to the matched token
#attrs = {"POS": "NOUN",'TAG':"HTGS"}
# Add rules to the attribute ruler
#ruler.add(patterns=patterns, attrs=attrs, index=0)  # "The" in "The Who"
#ruler.add(patterns=patterns, attrs=attrs, index=1)  # "Who" in "The Who"


	if case_insensitive:
		matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
	else:
		matcher = PhraseMatcher(nlp.vocab)
	return nlp,matcher






def remove_empty(list):
    return [x for x in list if not len(x)==0]

def common_post_mistake(w):
    #if w.text[0] in ['.']:
        #return 'PUNKT'
    if w.text in ['-','.','\n','/']:
        return 'PUNKT'
    elif w.text in ["l'","l’","d'","d’"]:
        return 'DET'
    elif w.text[0] in ['@','#']:
        return 'NOUN'
    else:
        return w.pos_

def compte_htgs(ws):
    #print("ws",ws,len(ws))
    htgs_num,sizes=0,[]
    for w in ws:
        sizes.append(len(w))
        if w.tag_=='HTG':
            htgs_num+=1
    return htgs_num,max(sizes)


# pattern_NP=[{"POS": "ADJ", "OP": "*"},{"POS": "NOUN", "OP": "+"}, {"POS": "ADJ", "OP": "*"}, {"POS": {'IN':["CC",'ADP','NUM','DET']}, "OP": "*"},{"POS": "ADJ", "OP": "*"},{"POS": "NOUN", "OP": "*"}]
# pattern_TW=[[{"TAG":"HTG","OP":"+"}],[{"POS": "ADJ", "OP": "*"},{"POS": "NOUN", "OP": "+"}, {"POS": "ADJ", "OP": "*"}, {"POS": {'IN':["CC",'ADP','NUM','DET']}, "OP": "*"},{"POS": "ADJ", "OP": "*"},{"POS": "NOUN", "OP": "*"}]]
# pattern_VG=[{"POS": "ADV", "OP": "*"},{"POS": "VERB", "OP": "+"},{"POS": "ADV", "OP": "*"}]
# pattern_HT=[{"TAG":"HTG","OP":"+"}]

pattern_HT=[{"TAG":"HTG","OP":"+"}]
pattern_NP=[{"POS": "ADJ", "OP": "*"},{"POS": {'IN':["NOUN",'PROPN']}, "OP": "+"}, {"POS": "ADJ", "OP": "*"}, {"POS": {'IN':["CC",'ADP','NUM','DET']}, "OP": "*"},{"POS": "ADJ", "OP": "*"},{"POS": {'IN':["NOUN",'PROPN']}, "OP": "*"}]
pattern_VG=[{"POS": "ADV", "OP": "*"},{"POS": "VERB", "OP": "+"},{"POS": "ADV", "OP": "*"}]

def sanity_chek(ws,remove_emoji=True,remove_http=True,remove_too_small=True):
	#remove EMOJIs from the list by default
	if remove_emoji:
		if ws._.has_emoji:
			return False
	#remove URLs from the list by default
	if remove_http:
		if 'http' in str(ws):
			return False
	if remove_too_small:
		if len(ws.text)<2:
			return False
	return True


remove_emoji=True
remove_http=True

def extract(corpus,lang='en',ngrmin=1,ngrmax=10,type_pattern=['NP'],NER_extract=False,remove_emoji=True,NER_types={"PER", "ORG", "GPE",'LOC'},starting=None):

	sample_dictionary={}
	sample_index={}

	pattern,authorizez_ending_tags=[],[]
	if 'NP' in type_pattern:
		pattern.append(pattern_NP)
		if lang=='en':
			authorizez_ending_tags.extend(['NOUN','PROPN'])
		else:
			authorizez_ending_tags.extend(['NOUN','PROPN','ADJ'])
	if 'VG' in type_pattern:
		pattern.append(pattern_VG)
		authorizez_ending_tags.extend(['VERB','ADV'])
	if 'HT' in type_pattern:
		pattern.append(pattern_HT)
		authorizez_ending_tags.extend(['NOUN','PROPN'])
	if NER_extract:
		authorizez_ending_tags.extend(['NOUN','PROPN'])
	authorizez_ending_tags=set(authorizez_ending_tags)

	if len(type_pattern)==0:
		if NER_extract:
			docs_terms = (textacy.extract.terms(doc,ents=partial(textacy.extract.entities,include_types=NER_types)) for doc in corpus)
	if len(type_pattern)>0:
		if NER_extract:
			docs_terms = (chain(textacy.extract.token_matches(doc,pattern),textacy.extract.terms(doc,ents=partial(textacy.extract.entities,include_types=NER_types))) for doc in corpus)
		else:
			docs_terms = (textacy.extract.token_matches(doc,pattern) for doc in corpus)
	nb_index=0

	for doc_id,doc in enumerate(docs_terms):
		already_seen_spans={}
		for ws in doc:
			if len(ws)>0:

				#first check that the pattern was not met before
				unique_signature=str(doc_id) + '_' + str(ws.start) +'_'  + str(ws.end)
				if unique_signature in already_seen_spans:
					pass
				else:
					already_seen_spans[unique_signature]=True

					#chech whether the starting character feature is met
					if starting==None:
						pass
					else:
						clause=False
						for start in starting:
							if ws.text[0][:len(start)]==start:
								clause=True
						if not clause:
							break


					#control for the POS of the last word
					#print('ws',ws)
					if  common_post_mistake(ws[-1]) in authorizez_ending_tags and sanity_chek(ws,remove_emoji=remove_emoji,remove_http=True,remove_too_small=True):

						ws_full=[]
						htgs,maxsize=compte_htgs(ws)#number of hashtags and max size of words in the multiterm
						if htgs<= 1 and maxsize>1:
							for w in ws:
								wl=w.lemma_.lower()
								#print (w,w.tag_)») and English quotation("") grammatically? If so what are they?
								if wl[-1] in '[!,\.:;"\'«»\-]' and w.tag_=='HTG':
									if len(wl)>=3:
										wl=wl[:-1]
								if wl[0] in '[!,\.:;"\'«»\-]':
									if len(wl)>=3:
										wl=wl[1:]
								ws_full.append(wl.replace('@','').replace('#',''))
							ws_full=remove_empty(ws_full)
							if len(ws_full)>=ngrmin and len(ws_full)<=ngrmax:
								ws_full.sort()
								ws_full_ordered=' '.join(ws_full)
								if len(ws_full_ordered)>1:
									if not ws_full_ordered in sample_dictionary:
										sample_dictionary[ws_full_ordered]={}
									wst=ws.text.strip()
									sample_dictionary[ws_full_ordered][wst]=sample_dictionary[ws_full_ordered].get(wst,0)+1
									sample_index.setdefault(ws_full_ordered,[]).append(doc_id)
									nb_index+=1


	print (len(sample_index),' candidate key terms extracted ')
	print (nb_index,' total occurrences ')
	return sample_dictionary,sample_index

def build_n_dict(sample_dictionary):
    n_dict={}
    for cle in sample_dictionary:
        n=len(cle.split())
        n_dict[cle]=n
    return(n_dict)

def build_nested(sample_dictionary,fast=False):

    n_dict={}
    n_dict_inv={}
    for cle in sample_dictionary:
        n=len(cle.split())
        n_dict[cle]=n
        n_dict_inv.setdefault(n,[]).append(cle)
    ns=list(n_dict_inv.keys())
    ns.sort()

    Nn=len(ns)
    cles=list(sample_dictionary.keys())
    cless={}
    for x in cles:
        cless[x]=set(x.split())

    nested={}
    for i in range(Nn-1):
        n=ns[Nn-i-1]
        nminus=ns[Nn-i-2]
        print ("finding nested strings for ngram of size:", n)
        larger=n_dict_inv[n]
        smaller=n_dict_inv[nminus]
        for x in tqdm(smaller,total=len(smaller)):
            xs=cless[x]
            for y in larger:
                if x in y:
                    if xs <= cless[y]:
                        nested.setdefault(x,[]).append(y)
                        if y in nested:
                            nested[x].extend(nested[y])
    return nested,n_dict

def pre_filtering(sample_dictionary,sample_index,threshold=2):
    sample_dictionary_filtered,sample_index_filtered={},{}
    for x in list(sample_dictionary.keys())[:]:
        N=sum(sample_dictionary[x].values())
        if N>=threshold:
            sample_dictionary_filtered[x] = sample_dictionary[x]
    for x in sample_index:
        if x in sample_dictionary_filtered:
            sample_index_filtered[x]=sample_index[x]
    return sample_dictionary_filtered,sample_index_filtered


def selecting(corpus,nested,sample_dictionary,sample_index,n_dict,freq_thres=.0001,topn=100,ranking='pigeon'):
	N=len(corpus)
	NW=corpus.n_tokens


	cpigeon,pigeon,freqn,cvalues,tfidf,freqb,docb={},{},{},{},{},{},{}
	print (len(sample_dictionary),' firsthand ')
	for w in sample_index:
	    d = len(set(sample_index[w]))
	    f = len(sample_index[w])
	    freqb[w]=f
	    docb[w]=d

	    fn = f / NW
	    dth = N - N * math.pow(float((N-1)/N),f)
	    #pigeon[w]=d/dth
	    if fn>freq_thres:
	        freqn[w]=fn
	        pigeon[w]=dth/d
	        tfidf[w]=f * math.log(N/d)
	        n=len(w)
	        if not w in nested:
	            cvalue=(math.log2(n)+.1)*f
	        else:
	            fnested=0
	            #print(nested[w])
	            for nested_term in nested[w]:
	                fnested+=len(sample_index[nested_term])
	            #print (freqn[w],nested[w],fnested,len(nested[w]),(math.log2(n)+0.1))
	            cvalue=(math.log2(n)+0.1)*(f-fnested/len(nested[w]))
	        cvalues[w]=cvalue
	        cpigeon[w]=cvalue*pigeon[w]

	if ranking=='pigeon':
	    print (len(pigeon),' short-listed after the frequency threshold filter ')
	    key_terms_list=list(map(lambda x:x[0],sorted(pigeon.items(),key=itemgetter(1),reverse=True)))[:topn]
	    print (len(key_terms_list),' short-listed after pigeon ')
	elif ranking=='C-value':
	    print (len(cvalues),' short-listed after the frequency threshold filter ')
	    key_terms_list=list(map(lambda x:x[0],sorted(cvalues.items(),key=itemgetter(1),reverse=True)))[:topn]
	    print (len(key_terms_list),' short-listed after c-value ')

	elif ranking=='tfidf':
	    print (len(tfidf),' short-listed after the frequency threshold filter ')
	    key_terms_list=list(map(lambda x:x[0],sorted(tfidf.items(),key=itemgetter(1),reverse=True)))[:topn]
	    print (len(key_terms_list),' short-listed after tfidf ')



	sample_dictionary_main={}
	for x in key_terms_list:
	    sample_dictionary_main[x]=sorted(sample_dictionary[x].items(),key=itemgetter(1),reverse=True)[0][0]

	word_list_df=pd.DataFrame(key_terms_list,columns=['lemma'])
	word_list_df['words']=word_list_df['lemma'].map(sample_dictionary)
	word_list_df['main_word']=word_list_df['lemma'].map(sample_dictionary_main)
	word_list_df['normalized frequency']=word_list_df['lemma'].map(freqn)
	word_list_df['documents']=word_list_df['lemma'].map(freqn)

	word_list_df['pigeon']=word_list_df['lemma'].map(pigeon)
	if ranking=='C-value' or ranking=='C-pigeon':
		word_list_df['C-value']=word_list_df['lemma'].map(cvalues)
		word_list_df['C-pigeon']=word_list_df['lemma'].map(cpigeon)
	else:
		word_list_df['weighted (ngr) frequency']=word_list_df['lemma'].map(cvalues)
		word_list_df['weighted (ngr) pigeon']=word_list_df['lemma'].map(cpigeon)

	word_list_df['tfidf']=word_list_df['lemma'].map(tfidf)
	word_list_df['n']=word_list_df['lemma'].map(n_dict)



	word_list_df['occurrences']=word_list_df['lemma'].map(freqb)
	word_list_df['documents']=word_list_df['lemma'].map(docb)


	#len(word_list_df)



	return word_list_df,sample_dictionary_main

def homogenize_wl(wl):
    lemma_2_wrds={}
    wrds_2_lemma={}
    for lemma,words in zip(wl['lemma'],wl['words']):
        #print('str(words)',str(words))
        #words=eval(words)
        try:
            words=eval(words)
        except:
            pass
        for w in words:
            if w in wrds_2_lemma:
                print (w,lemma)
                lemma=wrds_2_lemma[w]
                print('->',lemma)
        for w in words:
            wrds_2_lemma[w]=lemma
            if not lemma in lemma_2_wrds:
                lemma_2_wrds[lemma]={}
            lemma_2_wrds[lemma][w]=words[w]
    wl['words']=wl.lemma.map(lemma_2_wrds)
    wl=wl[wl['lemma'].isin(list(lemma_2_wrds.keys()))]

    try:
        new_main=[]
        for lemma,words,word_main in zip(wl['lemma'],wl['words'],wl['main_word']):
            try:
                sorted_words = sorted(eval(words).items(), key=lambda x: x[1], reverse=True)[0][0]
            except:
                sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[0][0]

            new_main.append(sorted_words)
        #if sorted_words!=word_main:
        #    print (lemma,words,word_main,sorted_words)
        wl['main_word']=new_main
    except:
        pass



    return wl

def substr_in_list(elem, lst):
  for s in lst:
    if elem != s and elem in s:
      return True
  return False

def sub_string(o,s):
    if o in s:
        os=set(o.split())
        ss=set(s.split())
        if os.issubset(ss):
            return True

def substring(string_list):
    string_list.sort(key=lambda s: len(s))
    out = []
    for s in string_list:
        if not any([sub_string(o,s) for o in out]):
            out.append(s)
    return out

def feed_matcher(nlp,matcher,sample_dictionary,sample_dictionary_main):#,case_insentive=True):
	minimal_query={}
	for cle in sample_dictionary_main:
		words=list(sample_dictionary[cle].keys())
		#if case_insentive:
		#	words=list(set(list(map(lambda x: x.lower(),words))))

		Nounphrases_list=substring(words)#[[j for j in i if not substr_in_list(j, i)] for i in words]

		#	words=sample_dictionary[cle]
		#	Nounphrases_list=substring(words)#[[j for j in i if not substr_in_list(j, i)] for i in words]

		minimal_query[cle]=Nounphrases_list
		#print ('Nounphrases_list',Nounphrases_list)
		patterns = [nlp.make_doc(Nounphrases) for Nounphrases in Nounphrases_list]
		#patterns =[['mr speaker']]
		#if 'speaker' in cle:
		#	print(cle,patterns)
		matcher.add(cle, patterns)
	return matcher, minimal_query


def index(summaries,sample_dictionary_main,nlp,matcher,name='default_name',exhaustive_export=False,type_export='simple',nested={}):
	if exhaustive_export:
		#index_df=pd.DataFrame(rows,columns=['term','lemma','actual form','doc_id','sent_id','start_id','end_id'])
		file_exhaustive=open('index_all'+name+'exhaustive.csv','w')
		file_exhaustive.write('\t'.join(['term','lemma','actual form','doc_id','sent_id','start_id','end_id'])+'\n')
	if type_export=='complex':
		#light_index_df=pd.DataFrame(lihgtrows,columns=['term','doc_id','sent_id','origin','source','date'])
		file=open('index_all'+name+'exhaustive.csv','w')
		file.write('"'+'"\t"'.join(['term','doc_id','sent_id','origin','source','date'])+'"\n')
	else:
		#light_index_df=pd.DataFrame(lihgtrows,columns=['term','doc_id','sent_id'])
		file=open('index_all'+name+'exhaustive.csv','w')
		file.write('\t'.join(['term','doc_id','sent_id'])+'\n')

	count={}
	count_doc={}
	rows=[]
	lihgtrows=[]
	forms_dict={}

	for i,couple in tqdm(enumerate(summaries),total=len(summaries)):
		#print(couple,i)
		if type_export=='complex':
			doc_id=couple[0]
			text=couple[1]
			origin=couple[2]
			source=couple[3]
			date=couple[4]
		else:
			text=couple
			doc_id=i
		#print (doc_id,text)
		#if case_insentive:
			#print(text)
		#	text=text.lower()
		doc=nlp(text.replace("’","'"))
		#print ('processing ',i)
		for sent_id,sent in enumerate(doc.sents):
			matches = matcher(sent)
			#if 'law'  in sent.text and 'practice' in sent.text:
			#print(sent.text)
			found_matches=[]
			for match_id, start, end in matches:
				found_matches.append(nlp.vocab.strings[match_id])
			found_matches=set(found_matches)
			for match_id, start, end in matches:

				#print ('we have a match')
				span = doc[start:end]
				match_id_string = nlp.vocab.strings[match_id]
				if len(set(nested.get(match_id_string,[])) & found_matches)==0:
					count_doc.setdefault(match_id_string,[]).append(doc_id)
					count[match_id_string]=count.get(match_id_string,0)+1
					#print(match_id_string)#,sample_dictionary_main[match_id_string])
					if not match_id_string in forms_dict:
						forms_dict[match_id_string]={}
					forms_dict[match_id_string][span.text]=forms_dict[match_id_string].get(span.text,0)+1
					if exhaustive_export:
						row=[sample_dictionary_main[match_id_string],match_id_string,span.text,doc_id,sent_id,start,end]
						file_exhaustive.write('\t'.join(list(map(lambda x:str(x),row)))+'\n')
					if type_export=='complex':
						lihgtrow=[sample_dictionary_main[match_id_string],doc_id,sent_id,origin,source,date]
						file.write('"'+'"\t"'.join(list(map(lambda x:str(x),lihgtrow)))+'"\n')
					elif type_export=='simple':
						lihgtrow=[sample_dictionary_main[match_id_string],doc_id,sent_id]
						file.write('\t'.join(list(map(lambda x:str(x),lihgtrow)))+'\n')
					else:
						pass
				#else:
				#	print ('no occ for', match_id_string, ' as ', found_matches, '\t',sent )
	if exhaustive_export:
		file_exhaustive.close()
	file.close()
	return count,count_doc,forms_dict



def hellinger2(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


sample_number=10

def build_potential_distributions(documents_sample,count_size,mapping={}):#actually mapping parameter is never used
	doc_len=[]
	for i,doc in enumerate(documents_sample):
		doc_len += [mapping.get(i,i)] * len(doc.split())
	potential_distributions={}
	for c in set(count_size.values()):
		for i in range(sample_number):
			potential_distributions.setdefault(c,[]).append(random.sample(doc_len, c))
	return potential_distributions


def dist_it(doc_list,mapping={}):
	d={}
	for i in doc_list:
		x=mapping.get(i,i)
		d[x]=d.get(x,0)+1/len(doc_list)#/size_dict.get(x,1)
	return d

def surprise(actual_list,size_dict,mapping={}):
	n=len(actual_list)
	total_words=sum(size_dict.values())
	#print (total_words)
	#print(size_dict)
	dict_prob_actual=dist_it(actual_list,mapping)#,size_dict,mapping)
	#print ('dict_prob_actual',sum(dict_prob_actual.values()))#,n/total_words,n)
	delta=0
	for i in dict_prob_actual:
		#print (i,dict_prob_actual[i],size_dict[i],n)
		delta+=dict_prob_actual[i]*math.log(dict_prob_actual[i]/(size_dict[i]/total_words))
	return delta

def compute_surprise(documents_sample,count_doc,mapping={}):
	count_doc_size={}#distribution of events size
	for w in count_doc:
		count_doc_size[w]=len(count_doc[w])
	potential_distributions=build_potential_distributions(documents_sample,count_doc_size)
	size_dict={}#distribution of (meta-)documents size
	for i,doc in enumerate(documents_sample):
		size_dict[mapping.get(i,i)]=size_dict.get(mapping.get(i,i),0)+len(doc.split())
	#print (size_dict)
	#random_prob_dict={}
	#for i,doc in enumerate(documents_sample):
	#	random_prob_dict[i]=1/size_dict[i]
	surprise_dict={}
	surprise_dict_random={}
	for word in tqdm(count_doc,total=len(count_doc)):
		surprise_random=[]
		for potential_distribution in potential_distributions[len(count_doc[word])]:
			#print ('potential_distribution',potential_distribution)
			surprise_random.append(surprise(potential_distribution,size_dict,mapping))#,potential_distributions[count[word]],len(documents_sample),mapping)
		norm=sum(surprise_random)/sample_number
		#if word=='year':
		#	print (word,surprise(count_doc[word],size_dict,mapping),norm)
		surprise_dict[word]=surprise(count_doc[word],size_dict,mapping)/norm#,potential_distributions[count[word]],len(documents_sample),mapping)
	return surprise_dict
