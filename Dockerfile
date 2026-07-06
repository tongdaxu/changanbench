FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install curl -y
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
COPY .condarc /root/.condarc
# RUN $HOME/miniconda3/bin/conda create -y -n illm python=3.10
# RUN $HOME/miniconda3/envs/illm/bin/pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements_illm.txt /tmp/requirements_illm.txt
RUN $HOME/miniconda3/bin/conda create -y -n illm python=3.10 && \
    $HOME/miniconda3/bin/conda run -n illm pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    $HOME/miniconda3/bin/conda run -n illm pip install -r /tmp/requirements_illm.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements_infinity.txt /tmp/requirements_infinity.txt
RUN $HOME/miniconda3/bin/conda create -y -n infinity python=3.10 && \
    $HOME/miniconda3/bin/conda run -n infinity pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    sed -i '/^torch==/d;/^torchvision==/d;/^torchaudio==/d;/^triton==/d;/^nvidia-/d;/flash_attn @/d' /tmp/requirements_infinity.txt && \
    $HOME/miniconda3/bin/conda run -n infinity pip install -r /tmp/requirements_infinity.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements_diffeic.txt /tmp/requirements_diffeic.txt
RUN $HOME/miniconda3/bin/conda create -y -n diffeic python=3.8 && \
    $HOME/miniconda3/bin/conda run -n diffeic pip install -r /tmp/requirements_diffeic.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements_stablecodec.txt /tmp/requirements_stablecodec.txt
RUN $HOME/miniconda3/bin/conda create -y -n stablecodec python=3.10 && \
    $HOME/miniconda3/bin/conda run -n stablecodec pip install -r /tmp/requirements_stablecodec.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /root
CMD ["/bin/bash"]