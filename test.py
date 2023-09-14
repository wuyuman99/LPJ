from transformers import AutoTokenizer, AutoModel
import jsonlines
import pandas as pd
from tqdm import tqdm
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
tokenizer = AutoTokenizer.from_pretrained("/home/wuyuman/chatglm-6B-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/wuyuman/chatglm-6B-int4", trust_remote_code=True).half().cuda()
model = model.eval()
# 载入数据

f = open("/home/wuyuman/Chatglm/data/CEs/elements.json",encoding='utf-8')
content = f.read()
elements = json.loads(content)
f = open("/home/wuyuman/Chatglm/data/articles/article_all.json",encoding='utf-8')
content = f.read()
articles = json.loads(content)

data=pd.read_csv('/home/wuyuman/Chatglm/data/retrivel_data/sbert_retrivel1.csv')
pre_promt = '根据案情陈述‘'
post_prompt = '’判断罪名。'
pred_crime_list = []


for i in tqdm(range(12265,len(data))):

    knowledge='已知法条规定如下。'
    # knowledge='已知：'
    retrievel_article=data['top_5_article'].iloc[i]
    for each_crime in eval(retrievel_article):
        knowledge+=(each_crime+'：'+articles[each_crime]['内容'])
        # knowledge+=(elements[each_crime[:-1]][0])
    fact=data['fact'].iloc[i]
    statement = knowledge + pre_promt + fact+ post_prompt
    if len(fact)>2000:
        statement = knowledge + pre_promt + fact[:2000]+ post_prompt
    
    response, history = model.chat(tokenizer, statement, history=[])
    pred_crime_list.append(response)
    
    with open("/home/wuyuman/Chatglm/article_retirvel7.json",'w',encoding='utf-8') as f:
        json.dump(pred_crime_list, f,ensure_ascii=False)





