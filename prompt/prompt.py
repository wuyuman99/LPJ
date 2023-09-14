import jsonlines
import json
import pandas as pd
data=pd.read_csv('/home/wuyuman/wuyuman/LPJ/data/retrivel_data/sbert_retrivel1.csv')
f = open("/home/wuyuman/wuyuman/LPJ/data/articles/article_all.json",encoding='utf-8')
content = f.read()
articles = json.loads(content)

file = open("/home/wuyuman/wuyuman/LPJ/prompt/fact_retrievel_top3.txt", 'w')   # 新文件
for i in range(len(data)):
    retrievel_article=data['top_5_article'].iloc[i]
    knowledge=''
    for each_crime in eval(retrievel_article):
        knowledge+=(each_crime+'：'+articles[each_crime]['内容'])
    # content1='请根据案情陈述“'+data['fact'].iloc[i]+'”判断：被告犯了什么罪？'
    # file.write(content1)
    # file.write('\n')  
    content2='已知部分法条规定如下。'+knowledge+'请根据案情陈述“'+data['fact'].iloc[i]+'”判断：被告犯了什么罪？'
    file.write(content2)   # 逐行写入
    file.write('\n')  
file.close()