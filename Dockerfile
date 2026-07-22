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

COPY wheels/pyiqa-0.1.16-py3-none-any.whl \
     /tmp/wheels/pyiqa-0.1.16-py3-none-any.whl

RUN /root/miniconda3/bin/conda run --no-capture-output -n diffeic \
    python -m pip install --no-deps \
    /tmp/wheels/pyiqa-0.1.16-py3-none-any.whl

COPY wheels/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl /tmp/wheels/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

RUN $HOME/miniconda3/bin/conda run -n infinity python -m pip install \
    /tmp/wheels/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY wheels/setuptools-81.0.0-py3-none-any.whl /tmp/wheels/setuptools-81.0.0-py3-none-any.whl

RUN $HOME/miniconda3/bin/conda run -n illm python -m pip install /tmp/wheels/setuptools-81.0.0-py3-none-any.whl

ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=20 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_PROGRESS_BAR=off

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /root/miniconda3/bin/conda create -y -n dcvc \
    python=3.8.20 pip=24.2 && \
    /root/miniconda3/bin/conda run -n dcvc python -m pip install \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 \
    matplotlib==3.5.3 numpy==1.21.6 Pillow==9.5.0 \
    pytorch-msssim==0.2.0 scipy==1.7.3 tqdm==4.66.1

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /root/miniconda3/bin/conda create -y -n dcvc_tcm \
    python=3.6.13 pip=21.2.2 && \
    /root/miniconda3/bin/conda run -n dcvc_tcm python -m pip install \
    --find-links https://download.pytorch.org/whl/torch_stable.html \
    torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0+cu111 \
    matplotlib==3.3.4 numpy==1.19.5 Pillow==8.4.0 \
    pytorch-msssim==0.2.0 scipy==1.5.4 tqdm==4.64.1

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /root/miniconda3/bin/conda create -y -n dcvc_hem \
    -c pytorch -c defaults \
    python=3.8.20 pip=24.2 \
    pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 && \
    /root/miniconda3/bin/conda run -n dcvc_hem python -m pip install \
    matplotlib==3.3.4 numpy==1.23.5 Pillow==9.5.0 \
    pytorch-msssim==0.2.0 scipy==1.10.1 tqdm==4.66.1

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /root/miniconda3/bin/conda create -y -n dcvc_dc \
    -c pytorch -c defaults \
    python=3.8.20 pip=24.2 \
    pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 && \
    /root/miniconda3/bin/conda run -n dcvc_dc python -m pip install \
    matplotlib==3.3.4 numpy==1.24.3 Pillow==10.4.0 \
    pytorch-msssim==0.2.0 scipy==1.10.1 tqdm==4.66.1

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /root/miniconda3/bin/conda create -y -n dcvc_fm \
    python=3.10.20 pip cmake ninja mkl=2023.1.0 && \
    /root/miniconda3/bin/conda run -n dcvc_fm python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 \
    matplotlib==3.8.4 numpy==1.26.4 Pillow==10.4.0 \
    pytorch-msssim==0.2.1 scipy==1.11.4 tqdm==4.67.1

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /root/miniconda3/bin/conda create -y -n dcvc_rt \
    python=3.12.13 pip && \
    /root/miniconda3/bin/conda run -n dcvc_rt python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
    matplotlib==3.10.3 numpy==1.26.4 Pillow==11.2.1 \
    pybind11==2.13.6 scipy==1.15.3 tqdm==4.67.1

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /root/miniconda3/bin/conda create -y -n dhvc \
    -c pytorch -c defaults \
    python=3.8.20 pip=24.2 \
    pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 && \
    /root/miniconda3/bin/conda run -n dhvc python -m pip install \
    compressai==1.2.6 einops==0.8.1 matplotlib==3.3.4 numpy==1.24.3 \
    Pillow==9.0.1 pybind11==2.13.6 pytorch-msssim==0.2.0 \
    safetensors==0.5.3 scikit-learn==1.3.2 scipy==1.10.1 \
    timm==1.0.3 tqdm==4.65.2

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /root/miniconda3/bin/conda create -y -n dcvc_b \
    -c pytorch -c defaults \
    python=3.8.20 pip=24.2 \
    pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 && \
    /root/miniconda3/bin/conda run -n dcvc_b python -m pip install \
    matplotlib==3.3.4 numpy==1.24.3 Pillow==10.4.0 \
    pytorch-msssim==0.2.0 scipy==1.10.1 tqdm==4.66.1

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /root/miniconda3/bin/conda create -y -n dcvc_sdd \
    -c pytorch -c defaults \
    python=3.8.20 pip=24.2 \
    pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 && \
    /root/miniconda3/bin/conda run -n dcvc_sdd python -m pip install \
    matplotlib==3.3.4 numpy==1.24.3 Pillow==10.4.0 \
    pytorch-msssim==0.2.0 scipy==1.10.1 tqdm==4.66.1

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /root/miniconda3/bin/conda create -y -n brhvc \
    python=3.10.20 pip cmake ninja mkl=2023.1.0 && \
    /root/miniconda3/bin/conda run -n brhvc python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 \
    matplotlib==3.8.4 numpy==1.26.3 Pillow==10.4.0 \
    pytorch-msssim==0.2.1 scipy==1.11.4 tqdm==4.67.1

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /root/miniconda3/bin/conda create -y -n vggt_env \
    python=3.10.20 pip && \
    /root/miniconda3/bin/conda run -n vggt_env python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    av==12.3.0 einops==0.8.2 huggingface-hub==0.34.4 lpips==0.1.4 \
    numpy==1.26.1 omegaconf==2.3.0 opencv-python-headless==4.11.0.86 \
    Pillow==11.3.0 pytorch-msssim==1.0.0 safetensors==0.7.0 tqdm==4.67.1 && \
    /root/miniconda3/bin/conda run --no-capture-output -n vggt_env \
    python -m pip install --no-deps \
    git+https://github.com/Wire-Byte/vggt_test.git@4c7eda13267348a29d91f9c6ea5564ef9000b7f2

COPY cab/models/dcvc_family/ /tmp/video-codec-sources/dcvc_family/
COPY cab/models/dhvc/ /tmp/video-codec-sources/dhvc/
COPY cab/models/dcvc_b/ /tmp/video-codec-sources/dcvc_b/
COPY cab/models/dcvc_sdd/ /tmp/video-codec-sources/dcvc_sdd/
COPY cab/models/kwai_nvc/ /tmp/video-codec-sources/kwai_nvc/
COPY docker/build_video_artifacts.sh /tmp/build_video_artifacts.sh

RUN bash /tmp/build_video_artifacts.sh && \
    rm -rf \
    /tmp/video-codec-sources \
    /tmp/build-* \
    /tmp/build_video_artifacts.sh && \
    /root/miniconda3/bin/conda clean --all --yes

ENV DCVC_PYTHON=/root/miniconda3/envs/dcvc/bin/python \
    DCVC_TCM_PYTHON=/root/miniconda3/envs/dcvc_tcm/bin/python \
    DCVC_HEM_PYTHON=/root/miniconda3/envs/dcvc_hem/bin/python \
    DCVC_DC_PYTHON=/root/miniconda3/envs/dcvc_dc/bin/python \
    DCVC_FM_PYTHON=/root/miniconda3/envs/dcvc_fm/bin/python \
    DCVC_RT_PYTHON=/root/miniconda3/envs/dcvc_rt/bin/python \
    DHVC_PYTHON=/root/miniconda3/envs/dhvc/bin/python \
    DCVC_B_PYTHON=/root/miniconda3/envs/dcvc_b/bin/python \
    DCVC_SDD_PYTHON=/root/miniconda3/envs/dcvc_sdd/bin/python \
    BRHVC_PYTHON=/root/miniconda3/envs/brhvc/bin/python \
    VIDEO_CODEC_ARTIFACT_ROOT=/opt/video_codec_artifacts \
    SUPPRESS_CUSTOM_KERNEL_WARNING=1

WORKDIR /root/changanbench

CMD ["/bin/bash"]
