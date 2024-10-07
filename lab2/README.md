---
jupyter:
  kernelspec:
    display_name: venv
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.5
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .markdown}
> 上面是对PPT内容的总结\
> 说一点自己的理解

-   Term-document matrices\
    假设一个 query 被分割为 m 个 term，而有 n 个待查询文档\
    这个矩阵的大小为 M \* N\
    对于元素 `[i][j]`则表示第`i`个 term 在第`j`个文档的的出现的频率\

-   bag of words model 词袋模型\
    不区分query的 term 顺序

-   Term frequency tf\
    \$ tf\_{t,d} \$为term `t` 在文档 `d`中出现的频率\

-   Log-frequency weighting\
    $$
    w_{t,d}=\begin{cases} 
    1+log_{10}tf_{t,d} & tf_{t,d} > 0 \\ 
    0 & \text otherwise 
    \end{cases}
    $$\

-   document frequency \$ df\_{t} \$ \$ df\_{t} \$ 就是对于 query 的一个
    term ，含有这个 term 的 document 数量\

-   idf\
    定义 N 为待查询文档的数目\
    那么 $$
    idf_t=log_{10}(N/df_t)
    $$

    > idf 反映了一个 term
    > 对查询文档的帮助，或者说这个term是否可以作为一个文档的`特性`，如果几乎所有文档都含有此
    > term，那么这个term就对查询没有什么帮助

-   **tf-idf weighting** $$
    w_{t,d}=log_{10}(1+tf_{t,d})*log_{10}(N/df_t)
    $$

    > 进一步，对query来说，对于一个query，我们可以得到任意文档d在此query
    > q上的score\
    > $$
    > Score(q,d)=\sum_{\text{t that both in q and d}}tf.idf_{t,d}
    > $$

-   使用向量\
    这里通俗的讲就是把 query 看作一个 document q，计算这个document
    q的vector和其他待查寻的document d的相似度，根据相似来排序\
    排序的方法有欧式距离和余弦相似度，使用**余弦相似度更合理**
:::

::: {.cell .markdown}
#### 总结一下实现的步骤应该是

-   传入一个查询\
-   对查询处理得到m个term\
-   计算得到一个 m\*n tf-idf矩阵(假设有n个待查询文档)\
-   把每一列作为一个document vector $v_i$，对于查询同样得到一个vector
    $v_o$
-   计算 $v_i$ 和 $v_o$的余弦相似度，最后可以得到一个rank\
-   输出
:::

::: {.cell .markdown}
### 导入库
:::

::: {.cell .code execution_count="20"}
``` python
import json  
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
import re  
import math
from tqdm import tqdm 
lemmatizer = WordNetLemmatizer()
legal_words=words.words()
```
:::

::: {.cell .markdown}
### 读取数据
:::

::: {.cell .code execution_count="21"}
``` python
tweets=[]
all_tweets_id=[]
f=open('/home/wangxv/Files/course/message_data/lab1/data/tweets.txt','r')
line_num=1
for line in f:
    tweets.append(json.loads(line))
    tweets[-1]['tweetId']=line_num  
    line_num+=1
    all_tweets_id.append(int(tweets[-1]['tweetId']))
f.close()
tweets=[{'id':i['tweetId'],'text':i['text']} for i in tweets]
```
:::

::: {.cell .markdown}
### 对文本进行预处理的函数
:::

::: {.cell .code execution_count="22"}
``` python
def deal_text(text:str):
    text=text.lower()#均转换为小写
    text = re.sub(r'[^a-z\s]', '', text)#仅保留字母和空格  
    tokens=word_tokenize(text)#获取分词结果  
    stop_words=set(stopwords.words('english'))
    filterd_tokens=[word for word in tokens if word not in stop_words]#去除停用词
    lemmatizer_tokens=[lemmatizer.lemmatize(word) for word in filterd_tokens]#还原词形
    #lemmatizer_tokens=[word for word in lemmatizer_tokens if word in legal_words]#只保留合法单词，加上这个跑得很慢
    return lemmatizer_tokens
```
:::

::: {.cell .markdown}
### 为tweets添加属性 标记处理后的text
:::

::: {.cell .code execution_count="24"}
``` python
from collections import defaultdict
for i in tqdm(range(len(tweets))):
    words=deal_text(tweets[i]['text'])
    word_dict=defaultdict(lambda: 0)
    for word in words:
        word_dict[word]+=1  
    tweets[i]['words']=word_dict  
```

::: {.output .stream .stderr}
    100%|██████████| 30364/30364 [00:20<00:00, 1499.14it/s]
:::
:::

::: {.cell .markdown}
### 构建一个单词-文档频率字典，加速之后的idf的计算(要不每次都要遍历30000多个文档，太慢了)
:::

::: {.cell .code execution_count="7"}
``` python
help_dict=defaultdict(lambda:0)
words=[]
#根据文档构建一个单词列表  
tweet_words=[]
for tweet in tqdm(tweets):
    words+=deal_text(tweet['text'])
    tweet_words.append(deal_text(tweet['text']))
words=list(set(words))
for word in tqdm(words):
    for tmp in tweet_words:
        if(word in tmp):
            help_dict[word]+=1
```

::: {.output .stream .stderr}
    100%|██████████| 30364/30364 [00:16<00:00, 1844.68it/s]
    100%|██████████| 53240/53240 [17:47<00:00, 49.85it/s]
:::
:::

::: {.cell .code}
``` python
#保存到文本文件
with open('help_dict.txt', 'w') as f:
    for key, value in help_dict.items():
        f.write(f"{key}:{value}\n")
```
:::

::: {.cell .code execution_count="25"}
``` python
help_dict = defaultdict(int)
with open('help_dict.txt', 'r') as f:
    for line in f:
        key, value = line.strip().split(':')
        help_dict[key] = int(value)
```
:::

::: {.cell .markdown}
### 计算 term-frequency
:::

::: {.cell .code execution_count="26"}
``` python
def get_tf(term,document):#传入term，document  
    return math.log10(1+document['words'][term])
```
:::

::: {.cell .code execution_count="27"}
``` python
def get_idf(term,documents):
    f=help_dict[term]  
    f=1 if f==0 else f  
    return math.log10(len(documents)/f)  
```
:::

::: {.cell .code execution_count="28"}
``` python
def get_weight(term,document,documents=tweets):
    return get_tf(term,document)*get_idf(term,documents)  
```
:::

::: {.cell .markdown}
### 定义根据查询获取排序的函数
:::

::: {.cell .code execution_count="71"}
``` python
def cos(v1,v2):
    up=0
    for ind in range(len(v1)):
        up+=v1[ind]*v2[ind] 
    down=1
    tmp=0  
    for e in v1:
        tmp+=e**2  
    down*=math.sqrt(tmp)
    tmp=0
    for e in v2:
        tmp+=e**2  
    down*=math.sqrt(tmp)
    if(down==0):
        return 0
    return math.fabs(up/down)
def retrieve(query,documents=tweets):
    terms=deal_text(query)
    document_vectors=[[0 for i in range(len(terms))] for u in range(len(documents))]
    for ind1 in tqdm(range(len(terms))):
        for ind2 in range(len(documents)):
            document_vectors[ind2][ind1]=get_weight(terms[ind1],documents[ind2])
    q_vector=[1 for i in terms]
    #计算角度   
    result=[(ind+1,cos(q_vector,document_vectors[ind])) for ind in range(len(document_vectors))]
    return sorted(result,key=lambda x:x[1],reverse=True)[:100]
```
:::

::: {.cell .code execution_count="85"}
``` python
query='machine learning'
```
:::

::: {.cell .code}
``` python
result=retrieve(query)
result
```
:::