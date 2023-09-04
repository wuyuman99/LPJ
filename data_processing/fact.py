import jsonlines
import json
import pandas as pd
all_crime =[]
with jsonlines.open("/home/wuyuman/Chatglm/data/fact_data/data_valid.json") as f:
    for line in f:
        all_crime.append(line)
f = open("/home/wuyuman/Chatglm/data/predict_crime/response_result.json", 'r')
content = f.read()
response_result = json.loads(content)

fact_list=[]
for i in range(len(all_crime)):   
    fact_list.append(all_crime[i]['fact'])
term_of_imprisonment=[]  
for i in range(len(all_crime)): 
    term_of_imprisonment.append(all_crime[i]['meta']['term_of_imprisonment'])
accusation=[]
for i in range(len(all_crime)): 
    accusation.append(all_crime[i]['meta']['accusation'])

death_penalty=[]
imprisonment=[]
life_imprisonment=[]
data=pd.DataFrame({'fact':fact_list})
for i in term_of_imprisonment:   
    death_penalty.append(i['death_penalty'])
    imprisonment.append(i['imprisonment'])
    life_imprisonment.append(i['life_imprisonment'])
data['accusation']=pd.DataFrame({'accusation':accusation})
data['death_penalty']=pd.DataFrame({'death_penalty':death_penalty})
data['imprisonment']=pd.DataFrame({'imprisonment':imprisonment})
data['life_imprisonment']=pd.DataFrame({'life_imprisonment':life_imprisonment})
data['crime_num']=[len(i) for i in data['accusation']]
data['predict_result']=response_result
data['accusation']=[' '.join(i) for i in data['accusation']]
# data.to_csv('all_data.csv')


def acc(data,pred_column1,pred_column2):
    acc_dict={}
    num_dict={}
    for i in data['accusation'].value_counts().index:
        data_new=data[data['accusation']==i]
        
        sum1=0
        sum2=0
        for j in range(len(data_new)):
            if data_new['accusation'].iloc[j] in data_new[pred_column1].iloc[j]:
                sum1+=1
            if data_new['accusation'].iloc[j] in data_new[pred_column2].iloc[j]:
                sum2+=1
        acc_dict[i]=sum1/len(data_new),sum2/len(data_new)
        num_dict[i]=len(data_new)
    return acc_dict