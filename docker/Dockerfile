# ubuntu 20.0 + cuda 11.4.2 + tensorRT 8.0.3.4 + cuDNN 8.2.4.15 + NCCL 2.11.4
# https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-10.html#rel_21-10
# the newest version for tensorrt8 + cuda 11.4.2
FROM  nvcr.io/nvidia/tensorrt:21.10-py3

WORKDIR /home/model-server

RUN apt-get update && \
    apt-get install -y --no-install-recommends vim python3-pip python3.8-dev python3-wheel openjdk-11-jdk locales language-pack-zh-hans language-pack-zh-hant && \
    apt-get install -y --no-install-recommends --upgrade libomp-dev libfreetype6 libnss3 libsasl2-2 libssl1.1 libsystemd0 libudev1 linux-libc-dev openssl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 可以考虑使用非 root 用户执行
# RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3
# ENV VIRTUAL_ENV=/opt/venv
# RUN python3.8 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY requirements.txt requirements.txt

RUN pip --no-cache-dir install --upgrade pip && \
    pip --no-cache-dir install https://paddle-inference-lib.bj.bcebos.com/2.3.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.3.2.post112-cp38-cp38-linux_x86_64.whl

RUN pip --no-cache-dir install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    rm /usr/share/python-wheels/*.whl && \
    rm -rf /root/.cache/pip/*

ENTRYPOINT ["tail", "-f", "/dev/null"]
