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
1. 基于 dockerfile 的镜像准备，进入 dockerfile 目录执行以下命令。
```commandline
docker build -t paddle-serving-base .
```


2. 模型准备

Paddle Serving 不管是 Pipeline 模式还是 C++ Serving 模式支持的模型结构都是 Paddle Serving Model 的形式，不管是什么类型的模型，都需要经过Model -> Inference Model -> Serving Model 的形式，才可以使用 Paddle Serving服务框架的形式启动起来，使用各类框架如何训练 Model，此处不做具体介绍。具体可以自行摸索。
![image.png](images/paddle-serving-map.png)

- Model 转换为 Inference Model

第一步是将模型转换为 Paddle Inference 可以使用的静态模型，具体的针对不同类型的模型，大致可以分为两种转换途径。

（1）Paddlepaddle 框架模型
如果是基于 PaddlePaddle 框架训练的模型，相对来说比较简单，可以直接使用paddle.jit 中的 to_static 工具进行转换，具体如下：
```python
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec
from paddlenlp.transformers.bert.modeling import BertForSequenceClassification

# 以 BERT 模型为例
paddle_model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
paddle_model.eval()

# 定义输入参数
input_ids = InputSpec([None, None], 'int64', 'input_ids')
token_type_ids = InputSpec([None, None], 'int64', 'token_type_ids')
attention_mask = InputSpec([None, None], 'int64', 'attention_mask')

# 导出静态模型
model = to_static(paddle_model, 
                  input_spec=[input_ids, token_type_ids, attention_mask])
paddle.jit.save(model, 'pb_model')
```
（2）Paddlepaddle 框架模型

如果是非 PaddlePaddle 框架模型，有两种方式：(a)如 Pytorch 模型，可以读取其参数，然后通过paddle_model.load_dict，将模型的转换为 paddlepaddle 模型，然后再进行正常的转换操作。（b）可以直接使用X2Paddle 工具将模型转换为 Inference Model的形式（一行代码即可）。
```python
# 以 onnx fp32的模型转换为例
from x2paddle.convert import onnx2paddle

onnx2paddle('pytorch_model.onnx', "inference_model")
```

- Inference Model转Paddle Serving Model

Inference Model 转为 Paddle Serving Model 相对还是比较简单的，只需要一行代码即可实现对应模型的转换，需要注意的一点是PaddlePaddle2.0之前模型和之后的模型结构不同，参数需要进行一些小的修改。
```python
import paddle_serving_client.io as serving_io

serving_io.inference_model_to_serving("serving_model",
                                      serving_server="serving_server", 
                                      serving_client="serving_client",
                                      model_filename='pb_model.pdmodel', 
                                      params_filename='pb_model.pdiparams')
```

3. 基于镜像启动服务

镜像分别开启了 RPC 的服务端口8088，以及 http 的服务端口18082，具体的代码可以参考：[https://github.com/yang9112/paddle-serving-experience](https://github.com/yang9112/paddle-serving-experience)
```
nvidia-docker run -itd \
        -p 39999:18082 \
        -p 39998:8088 \
        -v /paddle_workdir/serving_server:/home/model-server/serving_server \
        paddle-serving-base bash
```

# 2. 测试

# 3. 问题

# 4. 其他

## 4.1 关于运行镜像
官方提供的运行镜像实在有点大，因此基于 nvidia的官方镜像，重新打了一个适用于 Paddle Serving Pipeline 的镜像，相对来说要小点（10G左右）
具体[Dockerfile打包方案](https://github.com/yang9112/paddle-serving-experience/tree/main/docker)可以参考这里