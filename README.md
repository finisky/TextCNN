# TextCNN
A simple TextCNN pytorch implementation

实现基于：
[https://github.com/Shawn1993/cnn-text-classification-pytorch](https://github.com/Shawn1993/cnn-text-classification-pytorch)

主要改动：
* 简化了参数配置，希望呈现一个最简版本
* Fix一些由于pytorch版本升级接口变动所致语法错误
* Fix模型padding导致的runtime error
* 解耦模型model.py与training/test/prediction逻辑
* 定制tokenizer，默认中文jieba分词
* 使用torchtext的TabularDataset读取数据集：text \t label

使用的数据集是weibo_senti_100k中的部分数据，其中train/test分别有20000和3000条。

# Requirements
pytorch==1.3.1
torchtext==0.4.0

# Train
`python main.py -train`

# Test
`python main.py -test -snapshot snapshot/best_steps_400.pt`

运行结果：
```
Evaluation - loss: 0.061201  acc: 98.053% (2518/2568)
```

# Predict
`python main.py -predict -snapshot snapshot/best_steps_400.pt`

运行结果：
```
>>内牛满面~[泪]
0 | 内牛满面~[泪]
>>啧啧啧，好幸福好幸福
1 | 啧啧啧，好幸福好幸福
```