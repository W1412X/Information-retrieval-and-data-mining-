### 实验1 **Inverted index and Boolean Retrieval Model**  

#### 环境配置  
> `nltk`这个包或者`virtualenv`这个工具肯定有一个有bug()，我用nltk.data.path添加路径完全不管用    
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
  - 或者你直接运行python，然后执行
  ```python
  import nltk
  print(nltk.data.path)
  ```
  - 打印的路径就是你需要放进去的路径
#### 实现思路  
> `re.ipynb`里边注释挺全的  

#### 因为老师修改要求，所以test.py中是最终程序