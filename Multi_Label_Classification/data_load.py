import numpy as np
import pandas as pd
import os
from collections import Counter
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


##三级标签
grades = ['用户评论']
classes = ['商品种类','用户群体','评论情绪','评论标签']
categories = {  '商品种类':['服饰','数码产品','日用化妆品','食品',''],
                '用户群体':['儿童', '女性', '老年人'],
                '评论情绪':['好评','中评','差评']
              }



#将不同文件夹下的多个csv数据合并
root = r'.\data'
all_data = pd.DataFrame()

for grade in grades:
    for class_ in classes:
        for category in tqdm(categories[class_]):
            file = os.path.join(root,grade,class_,category+'.csv')
            data = pd.read_csv(open(file,encoding='utf8'))
            print('Size of data:',len(data))
            
            #按web-scraper-order排序
            data['web-scraper-order'] = data['web-scraper-order'].apply(lambda x: int(x.split('-')[1]))
            data = data[['web-scraper-order', 'item']]
            data = data.sort_values(by='web-scraper-order')
            
            data['item'] = data.item.apply(lambda x: ''.join(x.split()))
            data['labels'] = data.item.apply(lambda x: [grade,class_, category]+x[x.index('[商品标签：]')+6:].split(',') if x.find('[商品标签：]')!=-1 else [grade, class_, category])
            data['item'] = data.item.apply(lambda x:  x.replace('[评论]', ''))
            data['item'] = data.item.apply(lambda x:  x[:x.index('评价信息')] if x.index('评价信息') else x)
            
            data = data[['item','labels']]
            all_data = all_data.append(data)

print('Data Size:',len(all_data))


### 存在大量含标签的样本数量较少，为提高模型的分类效果，需要对样本进行过滤，将知识点较少的样本过滤掉

min_samples = 300  ###最小样本数量调整，可以获得含有不同数量标签的数据
    #减小样本数量，标签数量增加，可根据实际情况调整
    # 200   134
    # 100   228

data = all_data.copy()
labels = []
for i in data.labels:
    labels.extend(i)

##提取标签的唯一值 
result = dict(sorted(dict(Counter(labels)).items(), key=lambda x: x[1], reverse=True))
lens = np.array(list(result.values()))
LABEL_NUM = len(lens[lens > min_samples])
print('Label Num:',LABEL_NUM)

# 选定数据label
label_target = set([k for k, v in result.items() if v > min_samples])

## 保证 grade subject category 在前三位置
data['labels'] = data.labels.apply(
    lambda x: x[:3] + list(set(x) - set(x[:3]) & label_target))  
# 去除没有知识点的数据
data['labels'] = data.labels.apply(lambda x: None if len(x) < 4 else x)  
data = data[data.labels.notna()]

# 最终的labels数量
labels = []
[labels.extend(i) for i in data.labels]
LABEL_NUM = len(set(labels))
#打印数据量，标签数量
print(f'>{min_samples} datasize:{len(data)} multi_class:{LABEL_NUM}')

##保存数据，按照格式的不同保存
def save(data,type):
    if type=='csv':
        #将标签由‘ ’隔开
        data['labels'] = data.labels.apply(lambda x: ' '.join(x))
        
        # shuffle
        data = data.sample(frac=1)

        file = os.path.join(root, f'data_{LABEL_NUM}.csv')
        #保存数据
        data.to_csv(file, index=False, encoding='UTF8')  
        print('csv data file generated! ', file)
    elif type=='fasttext':
        # fasttext需要在标签前面添加__label__
        profix = '__label__'
        
        if profix:
            data['labels'] = data.labels.apply(lambda x: [profix + i for i in x])

        data['labels'] = data.labels.apply(lambda x: ' '.join(x))

        # shuffle
        data = data.sample(frac=1)

        file = os.path.join(root, f'data_{LABEL_NUM}{profix}.txt')

        with open(file, 'w',encoding='utf-8') as f:
            for index, row in data.iterrows():
                f.write(row['labels'] + ' ' + row['item'] + '\n')
        print('fasttext data file generated at ', file)
    else:
        print('Error Type!')

save(data,'csv')

