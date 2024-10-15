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

::: {.cell .code execution_count="2"}
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

::: {.cell .code execution_count="3"}
``` python
tweets=[]
all_tweets_id=[]
f=open('./data/tweets.txt','r')
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

::: {.cell .code execution_count="4"}
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

::: {.cell .code execution_count="5"}
``` python
from collections import defaultdict
#构建词汇表，方便之后的向量生成
global_words = set()
for i in tqdm(range(len(tweets))):
    words=deal_text(tweets[i]['text'])
    tweets[i]['length']=len(words)#添加文档长度信息
    global_words.update(words)#顺便构建全局词汇表
    word_dict=defaultdict(lambda: 0)
    for word in words:
        word_dict[word]+=1  
    tweets[i]['words']=word_dict  
global_words = list(global_words)  #转为列表方便索引
word_index = {word: i for i, word in enumerate(global_words)} #词到索引的映射
```

::: {.output .stream .stderr}
      0%|          | 0/30364 [00:00<?, ?it/s]
:::

::: {.output .stream .stderr}
    100%|██████████| 30364/30364 [00:11<00:00, 2697.68it/s]
:::
:::

::: {.cell .markdown}
### 构建一个单词-文档频率字典，加速之后的idf的计算(要不每次都要遍历30000多个文档，太慢了)

-   这里只需要运行一次，有了文件之后直接运行下一个就可获取help_dict
:::

::: {.cell .code}
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
#保存到文本文件
with open('help_dict.txt', 'w') as f:
    for key, value in help_dict.items():
        f.write(f"{key}:{value}\n")
```
:::

::: {.cell .code execution_count="19"}
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

::: {.cell .code execution_count="20"}
``` python
def get_tf(term,document):#传入term，document  
    return math.log10(1+document['words'][term])
```
:::

::: {.cell .code execution_count="21"}
``` python
def get_idf(term,documents):
    f=help_dict[term]  
    f=1 if f==0 else f  
    return math.log10(len(documents)/f)  
```
:::

::: {.cell .code execution_count="22"}
``` python
def get_weight(term,document,documents=tweets):#在这里添加长度惩罚
    return get_tf(term,document)*get_idf(term,documents)/document['length']
```
:::

::: {.cell .markdown}
### 定义根据查询获取排序的函数
:::

::: {.cell .code execution_count="26"}
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
def if_contain_key(words_key,words_dict):
    for key in words_key:
        if(words_dict[key]!=0):
            return True  
    return False 
# 检索函数
def retrieve(query, documents=tweets):
    #处理查询并构建查询向量
    terms = deal_text(query)
    query_vector = [0] * len(global_words)
    for term in tqdm(terms):
        if term in word_index:
            query_vector[word_index[term]] = 1 #这里设置为1，课上讲的简化
    #构建所有文档的向量，好吧，构架不了，这里文档数量*词汇量把我内存搞爆了，所以选择在这里直接找含有那个查询关键词的文档构建应该可以吧？
    #因为没有包含关键词的文档向量都是0了，没有区分度，排序也欸有意义
    selected_documents=[i for i in documents if if_contain_key(terms,i['words'])]
    document_vectors = [[0] * len(global_words) for _ in tqdm(range(len(selected_documents)))]
    for doc_id, document in tqdm(enumerate(selected_documents)):
        for term in document['words']:
            if term in word_index:
                document_vectors[doc_id][word_index[term]] = get_weight(term, document)
    #计算每个文档与查询的相似度
    results = [
        (selected_documents[ind]['id'], cos(query_vector, document_vectors[ind]))
        for ind in range(len(selected_documents))
    ]
    return sorted(results, key=lambda x: x[1], reverse=True)[:100]
```
:::

::: {.cell .code execution_count="32"}
``` python
query='machine learning'
result=retrieve(query)
print(result)
#然后这里对应输出的id和实际documents的索引是差了一个1(输出的是id是行号)
print(tweets[result[0][0]-1]['text'])
```

::: {.output .stream .stderr}
    100%|██████████| 2/2 [00:00<00:00, 16578.28it/s]
    100%|██████████| 49/49 [00:00<00:00, 5309.66it/s]
    49it [00:00, 35077.81it/s]
:::

::: {.output .stream .stdout}
    [(14902, 0.3379670540586379), (21503, 0.3264219283049397), (11257, 0.32232043134124416), (8416, 0.2749069700044688), (10992, 0.25602730412713154), (11288, 0.25412019167289196), (16603, 0.24576169010143106), (20215, 0.22941864832251613), (10648, 0.22536474647753965), (21435, 0.22324902030905638), (7397, 0.2209199661143098), (21991, 0.22004434050789773), (22005, 0.22004434050789773), (19609, 0.2150140872589951), (14973, 0.2146896348146268), (25760, 0.21211922167297903), (26040, 0.21211922167297903), (12052, 0.2118336622131355), (18696, 0.2112100574555666), (14288, 0.20994675618927755), (18517, 0.2098245403576975), (19038, 0.2094459689215885), (13517, 0.20907245946559092), (4271, 0.20817471708192126), (14750, 0.2076566312222904), (1962, 0.20593412384729715), (20956, 0.19714376830663594), (10529, 0.19688443845749468), (6696, 0.19471997887678816), (4250, 0.1937347757522228), (25442, 0.19263657646184373), (1371, 0.19238362705870585), (26275, 0.1920891124160952), (5375, 0.19154877471355714), (11391, 0.1914593681361676), (13527, 0.18870428911823708), (13624, 0.186745590214615), (13962, 0.186745590214615), (29042, 0.1801389756529235), (14046, 0.17973057424047229), (1069, 0.17872751829513994), (8415, 0.17791400499469165), (3477, 0.17710608874478623), (10320, 0.17463518924237745), (15170, 0.17358417828446704), (10593, 0.17148155798268136), (20149, 0.17111722924097464), (17406, 0.16672227069588694), (9642, 0.16025327021312796)]
    ^.^ my neighbor had a snow blower machine and did the whole sidewalk :)
:::
:::