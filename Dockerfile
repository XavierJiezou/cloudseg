# 使用一个基础的 Conda 镜像
FROM continuumio/miniconda3

# 将工作目录设置为 /app
WORKDIR /app

# 复制当前目录的内容到镜像的 /app 目录
COPY . /app

# 复制整个 Conda 环境到 Docker 镜像中
COPY ~/miniconda3/envs/cloudseg /opt/conda/envs/cloudseg

# 激活 Conda 环境并确保环境可用
RUN echo "source activate cloudseg" > ~/.bashrc
ENV PATH /opt/conda/envs/cloudseg/bin:$PATH

# 设置默认命令，进入bash并激活conda环境
CMD ["bash", "-c", "source activate cloudseg && exec bash"]
