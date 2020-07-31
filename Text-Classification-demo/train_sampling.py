file_name = r'./cnews_train.txt'
num_cat = {}
num_max = 1000
##从数据集中提取对不同label值的新闻，每个类别各取1000个
contents,labels = [],[]
with open(file_name,encoding='utf-8') as file:
    for line in file:
        ##逐行读取，并划分
        label,content = line.strip().split('\t')
        if content:
            if label not in num_cat:
                num_cat[label]=1
                contents.append(content)
                labels.append(label)
            else:
                #若该类别新闻数量不足1000，则继续提取
                if num_cat[label]<1000:
                    num_cat[label] = num_cat[label] + 1
                    contents.append(content)
                    labels.append(label)

#将采样后的数据写入文件
with open('cnews_small_sample.txt','w',encoding='utf-8') as f:
    for content,label in zip(contents,labels):
        f.write(label + '\t' + content+'\n')
    f.close()