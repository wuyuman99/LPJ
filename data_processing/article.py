import jsonlines
import json
import pandas as pd
# -*- coding: utf-8 -*-
import thulac
model=thulac.thulac(seg_only=True)

import pickle as pk
import numpy as np
all_crime =[]
with jsonlines.open("/home/wuyuman/Chatglm/data/fact_data/data_valid.json") as f:
    for line in f:
        all_crime.append(line)

accu=open('/home/wuyuman/Chatglm/data/new_accu.txt',encoding='gbk').readlines()
accu = [i.replace('\n', '') for i in accu]
f = open("/home/wuyuman/Chatglm/data/article.json", 'r')
content = f.read()
article = json.loads(content)

#然后CE文本是这两个：
CE1=open('/home/wuyuman/Chatglm/data/CEs/CEs.txt',encoding='gbk').readlines()
CE2=open('/home/wuyuman/Chatglm/data/CEs/CEs_supp.txt').readlines()
CE=CE1+CE2
charge_num=len(accu)
ele_dict={}

for charge_id in range(charge_num):
    charge=accu[charge_id]

    if charge=='走私普通货物、物品':
        charge='走私普通货物物品'
    for ce in CE:  #从CE集合中，找这个罪名对应的CE
        if ce.startswith(charge):
            #这就是那个CE
            #将CE拆为罪名和4个部分
            parts=[x.strip() for x in ce.split('&')]
            #5或6个元素。第一个元素是罪名，然后倒数4个分别是subject-subjective-object-objective
            ele_subject=parts[-4]
            ele_subjective=parts[-3]
            ele_object=parts[-2]
            ele_objective=parts[-1]
            definition=''.join(parts[:-4])
    
            ele_dict[charge]=[definition,ele_subject,ele_subjective,ele_object,ele_objective]
with open("/home/wuyuman/Chatglm/elements.json",'w',encoding='utf-8') as f:
        json.dump(ele_dict, f,ensure_ascii=False)


article_dict={}
# 有第××条之一
for i in range(1,len(article)):
    content=article[i]
    article_num=content.split('\u3000')[0]
    crime=(content.split('】')[0]).split('【')[1]

    # 只有一个【】
    if len(content.split('】'))==2:
        crime_description=content.split('】')[1:]
        crime_description=''.join(crime_description)
        crime_list=crime.split(';')
        crime_description_list=crime_description.split('\u3000')
        #单罪名
        if len(crime.split(';'))==1:
            crimes=crime.split('罪、')
            for each_crime in crimes:
                if each_crime[-1]!='罪':
                    each_crime=each_crime+'罪' 
                if each_crime not in article_dict :
                    article_dict[each_crime]={'法条':article_num,'内容':crime_description}
                else:
                    article_dict[each_crime]={'法条':article_dict[each_crime]['法条']+'&'+article_num,'内容':article_dict[each_crime]['内容']+'&'+crime_description}       
        # 多个罪名且有之一
        elif len(crime_list)>1 and '\u3000' in crime_description:
            article_dict[crime_list[0]]={'法条':article_num,'内容':''.join(crime_description.split('\u3000')[0].split('。')[:-1])}
            if len(crime_list)==2:
                article_dict[crime_list[1]]={'法条':crime_description.split('\u3000')[0].split('。')[-1],'内容':crime_description_list[-1]}
            # else:
            #     print(crime)
            #     print(content)
        #多个罪名且无之一
        elif len(crime_list)>1 and '\u3000' not in crime_description:
            for crime in crime_list:
                if crime not in article_dict:
                    article_dict[crime]={'法条':article_num,'内容':crime_description}
                else:
                    article_dict[crime]={'法条':article_dict[crime]['法条']+'&'+article_num,'内容':article_dict[crime]['内容']+'&'+crime_description}
                
    if len(content.split('】'))>2:
        # print(content)
        crime_description=''.join(content.split('】')[1:])
        crime_description=crime_description.split('【')[0]
        for crime in ((content.split('】')[0]).split('【')[1]).split(';'):
            article_dict[crime]={'法条':article_num,'内容':crime_description}
        # print(crime_description)
        # print(article_dict[crime])
        crime_add=(content.split('【')[-1]).split('】')[0]
        crime_description_add=content.split('】')[-1]
        article_dict[crime_add]={'法条':article_num,'内容':crime_description_add}

for each_article in article_dict:
    if '&' in article_dict[each_article]['法条']:
        article_num_list=article_dict[each_article]['法条'].split('&')
        article_description_list=article_dict[each_article]['内容'].split('&')
        for i in range(len(article_num_list)):
            if article_num_list[i] in article_dict[each_article]['内容']:
                article_dict[each_article]['法条']=article_num_list[i]
                article_dict[each_article]['内容']=article_description_list[i]
                print(each_article)
                print(article_dict[each_article])

with open("/home/wuyuman/Chatglm/article_all.json",'w',encoding='utf-8') as f:
        json.dump(article_dict, f,ensure_ascii=False)
