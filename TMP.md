### Install Docker  
- 我的系统是 ubuntu 20.04    
- 添加密钥  
```shell
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```
- 添加docker仓库  
  - 打开文件 /etc/apt/sources.list.d/docker.list(没有的话直接创建)
  - 添加以下字段 
  ```shell
  deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu/ focal stable
  ```
  - 注意替换focal为ubuntu的版本代号，可以使用echo $(lsb_release -cs) 获取
  - 注意电脑的架构，修改arch= 参数  
  - 切换其他源可以自己去找，这里用的阿里镜像 
- 检查是否添加成功  
  - sudo apt update  
  - 没有报错就是成功  
- 安装  
	```shell  
	sudo apt-get install docker-ce docker-ce-cli containerd.io
	``` 
  
- 拉取镜像网络问题
### 配置Docker镜像  
> 镜像站列表  
https://xuanyuan.me/blog/archives/1154  
```shell
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json

 <<-'EOF'
{
    "registry-mirrors": [
        "https://docker.rainbond.cc",
        "https://docker.udayun.com/",
        "https://docker.211678.top"
    ]
}

EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```