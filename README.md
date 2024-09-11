# 山东大学信息检索实验  
## 编程语言 `python 3.11.5`
### 实验1 **Inverted index and Boolean Retrieval Model**  

#### 环境配置  
> `nltk`这个包或者`virtualenv`这个工具肯定有一个有bug，我添加nltk.data.path添加路径完全不管用  
> **最好还是放到你当前python环境的share文件夹里**  
- 安装nltk (下载慢使用清华源安装)
- 获取nltk数据
  - 首先从github获取nltk_data
  ```shell
  git clone https://github.com/nltk/nltk_data.git
  ```
  - 之后直接将得到的nltk文件夹中的packages更名为nltk_data
  - 将nltk_data放到python环境的share文件夹内  
  - 例如我的就是 `./venv/share/` 中  
#### 实现思路  
> 倒排索引构建  
- 收集构建索引的文档集
- 文本预处理，即分词，去除停用词，词形还原(英文running->run)以及去除特殊符号  
- 构建词典，包含所有的唯一术语的列表。  
- 生成倒排列表
  - 记录术语出现的位置(文档ID)
  - 更新倒排列表  
> 代码
#### 实现代码  
`./lab1/*`  
