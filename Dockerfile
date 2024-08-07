# 使用一个基础的 Conda 镜像
FROM continuumio/miniconda3

# 复制环境文件到镜像中
COPY environment.yml /tmp/environment.yml

# 创建 Conda 环境
RUN conda env create -f /tmp/environment.yml

# 激活 Conda 环境并确保环境可用
RUN echo "source activate cloudseg" > ~/.bashrc
ENV PATH /opt/conda/envs/cloudseg/bin:$PATH

# 将工作目录设置为 /app
WORKDIR /app

# 复制当前目录的内容到镜像的 /app 目录
COPY . /app

# 设置默认命令
CMD ["python", "-c 'import torch; print(torch.cuda.is_available())'"]