# Paddle-Serving-Experience

> 之前使用过 tensorflow serving、torchserve 以及 FastAPI 的模型部署方式，考虑到还未践行过国产框架的部署实践。本文主要从服务框架的使用者角度，记录了实现paddle模型的部署过程，以及在使用其他类型的模型（比如 ONNX）相关模型中遇到的一些问题以及解决办法。

# 1. 部署
## 1.1 环境准备
> 此处环境主要使用docker镜像进行部署与测试，具体dockerfile环境见4.1节，

基础环境：ubuntu 20.0 + cuda 11.4.2 + tensorRT 8.0.3.4 + cuDNN 8.2.4.15 + NCCL 2.11.4
Paddle Serving 环境：
```
paddle-bfloat             0.1.7
paddle-serving-app        0.9.0
paddle-serving-client     0.9.0
paddle-serving-server-gpu 0.9.0.post112
paddle2onnx               1.0.0
paddlefsl                 1.1.0
paddlenlp                 2.4.0
paddlepaddle-gpu          2.3.2.post112
```

## 1.2 服务部署
1. 基于 dockerfile 打包镜像
```commandline
docker build -t paddle-serving-base .
```
2. 基于镜像启动服务
```commandline

```

# 2. 测试

# 3. 问题

# 4. 其他

## 4.1 关于运行镜像
官方提供的运行镜像实在有点大，因此基于 nvidia的官方镜像，重新打了一个适用于 Paddle Serving Pipeline 的镜像，相对来说要小点（10G左右）
具体[Dockerfile打包方案](https://github.com/yang9112/paddle-serving-experience/tree/main/docker)可以参考这里