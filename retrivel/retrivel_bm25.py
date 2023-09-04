from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import scipy
from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd
import re
from tqdm import tqdm
import jieba

f = open("/home/wuyuman/Chatglm/elements.json",encoding='utf-8')
content = f.read()
elements = json.loads(content)
f = open("/home/wuyuman/Chatglm/article_all.json",encoding='utf-8')
content = f.read()
articles = json.loads(content)
data=pd.read_csv('/home/wuyuman/Chatglm/test_data_all.csv')
accu=open('/home/wuyuman/Chatglm/data/new_accu.txt',encoding='gbk').readlines()
accu = [i.replace('\n', '') for i in accu]

all_data=pd.read_csv('/home/wuyuman/Chatglm/fact_selected.csv')
f = open("/home/wuyuman/Chatglm/article_selected.json",encoding='utf-8')
content = f.read()
article_selected = json.loads(content)

article_content=[]
crime_definition=[]
definition_and_article=[]
crimes=[]

for accusation in article_selected:
    article_content.append(accusation+':'+article_selected[accusation]['内容'])
    crime_definition.append(elements[accusation[:-1]][0])
    definition_and_article.append(elements[accusation[:-1]][0]+article_selected[accusation]['内容'])
    crimes.append(accusation)

stopwords = open('/home/wuyuman/Chatglm/hit_stopwords.txt').read().split('\n')
def clean_token(sentence_tokens):
    tokens = []
    for word in sentence_tokens:
      if word in stopwords:
        continue
      else:
        tokens.append(word)
    return tokens

def bm25_retrivel(fact_data,knowledge,if_clean):
    data=fact_data.copy()
    doc_list = [doc for doc in knowledge if doc != '']
    tokenized_corpus = []
    retrivel_top10=[]
    retrivel_top5=[]
    retrivel_top3=[]
           
    for sentence in doc_list:
        sentence_words = jieba.lcut(sentence)
        if if_clean==1:
            tokenized_corpus.append(clean_token(sentence_words))
        else:
            tokenized_corpus.append(sentence_words)

    bm25 = BM25Okapi(tokenized_corpus)
    for query in tqdm(data['fact']):
        tokenized_query = jieba.lcut(query)
        if if_clean==1:
            tokenized_query=clean_token(tokenized_query)
        retrivel_top10.append([''.join(i) for i in bm25.get_top_n(tokenized_query, tokenized_corpus, n=10)])
        retrivel_top5.append([''.join(i) for i in bm25.get_top_n(tokenized_query, tokenized_corpus, n=5)])
        retrivel_top3.append([''.join(i) for i in bm25.get_top_n(tokenized_query, tokenized_corpus, n=3)])
    data['bm25_top5']=[[j.split('罪')[0]+'罪' for j in i]  for i in retrivel_top5]
    data['bm25_top10']=[[j.split('罪')[0]+'罪' for j in i]  for i in retrivel_top10]
    data['bm25_top3']=[[j.split('罪')[0]+'罪' for j in i]  for i in retrivel_top3]

    return data

def retrievel_acc(data,column1,column2,column3):
    acc1=0
    acc2=0
    acc3=0
    for i in range(len(data)):
        if data['accusation'].iloc[i] in data[column1].iloc[i]:
            acc1+=1
        if data['accusation'].iloc[i] in data[column2].iloc[i]:
            acc2+=1
        if data['accusation'].iloc[i] in data[column3].iloc[i]:
            acc3+=1
    return column1+':'+str(acc1),column2+':'+str(acc2),column3+':'+str(acc3)

def each_acc(data,column):
    acc_dict={}
    num_dict={}
    for i in data['accusation'].value_counts().index:
        data_new=data[data['accusation']==i]
        sum=0
        for j in range(len(data_new)):
            if data_new['accusation'].iloc[j] in data_new[column].iloc[j]:
                sum+=1
        acc_dict[i]=sum/len(data_new)
    return acc_dict

data1=bm25_retrivel(fact_data=all_data,knowledge=article_content,if_clean=1)
data2=bm25_retrivel(fact_data=all_data,knowledge=crime_definition,if_clean=1)
data3=bm25_retrivel(fact_data=all_data,knowledge=definition_and_article,if_clean=1)

data10=bm25_retrivel(fact_data=all_data,knowledge=article_content,if_clean=0)
data20=bm25_retrivel(fact_data=all_data,knowledge=crime_definition,if_clean=0)
data30=bm25_retrivel(fact_data=all_data,knowledge=definition_and_article,if_clean=0)

data1.to_csv('/home/wuyuman/Chatglm/data/retrivel_data/bm25_article.csv')
data2.to_csv('/home/wuyuman/Chatglm/data/retrivel_data/bm25_crime.csv')
data3.to_csv('/home/wuyuman/Chatglm/data/retrivel_data/bm25_article_and_crime.csv')
data1.to_csv('/home/wuyuman/Chatglm/data/retrivel_data/bm25_article_stop.csv')
data2.to_csv('/home/wuyuman/Chatglm/data/retrivel_data/bm25_crime_stop.csv')
data3.to_csv('/home/wuyuman/Chatglm/data/retrivel_data/bm25_article_and_crime_stop.csv')