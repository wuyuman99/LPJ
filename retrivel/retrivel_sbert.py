from sentence_transformers import SentenceTransformer
import scipy
from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd
import re
from tqdm import tqdm

# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

f = open("/home/wuyuman/Chatglm/data/CEs/elements.json",encoding='utf-8')
content = f.read()
elements = json.loads(content)
f = open("/home/wuyuman/Chatglm/data/articles/article_all.json",encoding='utf-8')
content = f.read()
articles = json.loads(content)
data=pd.read_csv('/home/wuyuman/Chatglm/test_data_all.csv')


all_data=pd.read_csv('/home/wuyuman/Chatglm/data/predict_crime/all_data.csv')
all_data=all_data[all_data['crime_num']==1]
all_data['accusation']=[re.sub('[][]', '', i)+'罪' for i in all_data['accusation']]
accusation_list=all_data['accusation'].drop_duplicates()

accusation_selected=[]
for accusation in accusation_list.tolist():
    if accusation in articles.keys() and accusation in [i+'罪' for i in elements.keys()]:
        accusation_selected.append(accusation)

all_data=all_data[all_data['accusation'].isin(accusation_selected)]
all_data=all_data[['fact', 'accusation', 'death_penalty', 'imprisonment','life_imprisonment', 'crime_num', 'predict_result']]
# all_data.to_csv('fact_selected.csv')

article_selected={}
for accusation in articles:
    if accusation in accusation_selected:
        article_selected[accusation]=articles[accusation]
# with open("/home/wuyuman/Chatglm/article_selected.json",'w',encoding='utf-8') as f:
#         json.dump(article_selected, f,ensure_ascii=False)

def retrivel(fact_data,model,article_selected,elements):
    data=fact_data.copy()
    article_retrievel_top10=[]
    article_retrievel_top5=[]
    article_retrievel_top3=[]
    crime_retrievel_top10=[]
    crime_retrievel_top5=[]
    crime_retrievel_top3=[]
    definition_and_article_retrievel_top10=[]
    definition_and_article_retrievel_top5=[]
    definition_and_article_retrievel_top3=[]

    article_content=[]
    crime_definition=[]
    definition_and_article=[]
    crimes=[]

    for accusation in article_selected:
        article_content.append(accusation+':'+article_selected[accusation]['内容'])
        crime_definition.append(elements[accusation[:-1]][0])
        definition_and_article.append(elements[accusation[:-1]][0]+article_selected[accusation]['内容'])
        crimes.append(accusation)
    article_representation=model.encode(article_content)
    crime_definition_representation=model.encode(crime_definition)
    definition_and_article_representation=model.encode(definition_and_article)

    for i in tqdm(range(len(data))):
        fact_representation=model.encode(all_data['fact'].iloc[i])

        correlation1={}
        correlation2={}
        correlation3={}
        for j in range(len(article_content)):
            correlation1[crimes[j]]= util.cos_sim(fact_representation,article_representation[j])
            correlation2[crimes[j]]= util.cos_sim(fact_representation,crime_definition_representation[j])
            correlation3[crimes[j]]= util.cos_sim(fact_representation,definition_and_article_representation[j])
        article_retrievel_top10.append(sorted(correlation1.items(),key = lambda correlation1:correlation1[1],reverse=True)[:10])
        article_retrievel_top5.append(sorted(correlation1.items(),key = lambda correlation1:correlation1[1],reverse=True)[:5])
        article_retrievel_top3.append(sorted(correlation1.items(),key = lambda correlation1:correlation1[1],reverse=True)[:3])
        crime_retrievel_top10.append(sorted(correlation2.items(),key = lambda correlation2:correlation2[1],reverse=True)[:10])
        crime_retrievel_top5.append(sorted(correlation2.items(),key = lambda correlation2:correlation2[1],reverse=True)[:5])
        crime_retrievel_top3.append(sorted(correlation2.items(),key = lambda correlation2:correlation2[1],reverse=True)[:3])
        definition_and_article_retrievel_top10.append(sorted(correlation3.items(),key = lambda correlation3:correlation3[1],reverse=True)[:10])
        definition_and_article_retrievel_top5.append(sorted(correlation3.items(),key = lambda correlation3:correlation3[1],reverse=True)[:5])
        definition_and_article_retrievel_top3.append(sorted(correlation3.items(),key = lambda correlation3:correlation3[1],reverse=True)[:3])

    data['top_5_article']=[[i[j][0] for j in range(len(i))] for i in article_retrievel_top5]
    data['top_3_article']=[[i[j][0] for j in range(len(i))] for i in article_retrievel_top3]
    data['top_10_article']=[[i[j][0] for j in range(len(i))] for i in article_retrievel_top10]
    data['top_5_crime']=[[i[j][0] for j in range(len(i))] for i in crime_retrievel_top5]
    data['top_3_crime']=[[i[j][0] for j in range(len(i))] for i in crime_retrievel_top3]
    data['top_10_crime']=[[i[j][0] for j in range(len(i))] for i in crime_retrievel_top10]
    data['top_5_definition_and_article']=[[i[j][0] for j in range(len(i))] for i in definition_and_article_retrievel_top5]
    data['top_3_definition_and_article']=[[i[j][0] for j in range(len(i))] for i in definition_and_article_retrievel_top3]
    data['top_10_definition_and_article']=[[i[j][0] for j in range(len(i))] for i in definition_and_article_retrievel_top10]
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



model1 = SentenceTransformer('distiluse-base-multilingual-cased-v1')
model2 = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model3=SentenceTransformer('/home/wuyuman/hfl/chinese-roberta-wwm-ext')

data1=retrivel(fact_data=all_data,model=model1,article_selected=article_selected,elements=elements)
data2=retrivel(fact_data=all_data,model=model2,article_selected=article_selected,elements=elements)
data3=retrivel(fact_data=all_data,model=model3,article_selected=article_selected,elements=elements)
data1.to_csv('sbert_retrivel1.csv')
data2.to_csv('sbert_retrivel2.csv')
data3.to_csv('sbert_retrivel3.csv')
