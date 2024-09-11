import json  
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
import re
from tqdm import tqdm 
lemmatizer = WordNetLemmatizer()
legal_words=words.words()
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
def deal_text(text:str):
    text=text.lower()#均转换为小写
    text = re.sub(r'[^a-z\s]', '', text)#仅保留字母和空格  
    tokens=word_tokenize(text)#获取分词结果  
    stop_words=set(stopwords.words('english'))
    filterd_tokens=[word for word in tokens if word not in stop_words]#去除停用词
    lemmatizer_tokens=[lemmatizer.lemmatize(word) for word in filterd_tokens]#还原词形
    #lemmatizer_tokens=[word for word in lemmatizer_tokens if word in legal_words]#只保留合法单词，加上这个跑得很慢
    return lemmatizer_tokens
sign_sequence=[]
for tweet in tqdm(tweets):
    words=[]
    words+=deal_text(tweet['userName'])
    words+=deal_text(tweet['text'])
    for word in words:
        sign_sequence.append((word,int(tweet['tweetId'])))
unique_sign_sequence=list(set(sign_sequence))
unique_sign_sequence.sort(key=lambda x:(x[0],x[1]))
keyword_dict={}
for word in unique_sign_sequence:
    if(word[0] in keyword_dict):
        keyword_dict[word[0]].append(word[1])
    else:
        keyword_dict[word[0]]=[word[1]]
def op_and(list1,list2):
    result=[]
    if(list1==[] or list2==[]):
        return result 
    else:
        p1=0
        p2=0
        while(True):
            if(p1==len(list1) or p2==len(list2)):
                break 
            if(list1[p1]==list2[p2]):
                result.append(list1[p1])
                p1+=1
                p2+=1
            else:
                if(list1[p1]>=list2[p2]):
                    p2+=1
                else:
                    p1+=1
        return result
def op_or(list1,list2):
    result=[]
    if(list2==[] and list1==[]):
        return result 
    else:
        p1=0
        p2=0
        while(True):
            if(p1==len(list1) or p2==len(list2)):
                break  
            if(list1[p1]>=list2[p2]):#如果第一个比第二个大，先放第二个
                result.append(list2[p2])
                p2+=1
            else:
                result.append(list1[p1])
                p1+=1
        result+=list1[p1:]
        result+=list2[p2:]
        return result
def op_not(list0,list1=all_tweets_id):
    if(list0==[]):
        return list1
    else:
        return [i for i in list1 if i not in list0]
def op_and_not(list1,list2):
    result=[]
    if(list1==[]):
        return result  
    else:
        p1=0
        p2=0
        while(p1<len(list1)):
            id=list1[p1]
            p1+=1  
            while(p2<len(list2) and list2[p2]<id):
                p2+=1
            if(p2<len(list2) and list2[p2]==id):
                continue 
            else:
                result.append(id)
        return result
def deal_word(word:str):
    word=word.lower()
    word=re.sub(r'[^a-z]','',word)
    word=lemmatizer.lemmatize(word)
    return word
def op(pre,beh,type):
    if(type=='and'):
        return op_and(pre,beh)
    elif(type=='or'):
        return op_or(pre,beh)
    elif(type=='not'):#这个需要反过来
        return op_not(beh,pre)
    elif(type=='andnot'):
        return op_and_not(pre,beh)
while(True):
    try:
        query=input('input the query >>')
        queries=query.split(' ')
        if(deal_word(queries[0])=='not'):
            l=len(queries)
            result=op_not(keyword_dict[deal_word(queries[1])])
            for i in range(1,l//2):
                ind=2*i
                beh_list=keyword_dict[deal_word(queries[ind+1])]
                result=op(result,beh_list,queries[ind])
        else:
            l=len(queries)
            result=keyword_dict[deal_word(queries[0])]
            for i in range((l//2)):
                ind=2*i+1  
                beh_list=keyword_dict[deal_word(queries[ind+1])]
                result=op(result,beh_list,queries[ind])
        print(result)
    except Exception as e:
        print('程序异常')
        pass 