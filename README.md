# AI基准测试

## 1. 目录结构
包含2个训练和2个推理任务
```
- aibench
    ├──image_classification_train       // 图像分类训练任务
    ├──image_classification_inference   // 图像分类推理任务
    ├──time_series_forecasting_train    // 时间序列预测训练任务
    ├──speech_recognition_inference     // 语音识别推理任务
```

## 2. 安装环境
创建conda环境
```
conda create -n aibench python=3.9
```
安装依赖项
```
pip3 install requirements.txt
```
**Note: speech_recognition_inference任务要求具有make**

## 3. 部署
进入某个任务目录，如：
```
cd image_classification_train
```

任意一个任务目录下有`prepare.py`和`run.py`两个文件
```
- aibench
    ├──any_task         // 某个任务
        ├──prepare.py   // 数据集、模型的下载和预处理
        ├──run.py       // 运行基准测试
    ...
```
#### 3.1 准备工作

数据集、模型的下载和预处理
```
python prepare.py
```
准备工作有多个阶段，如数据集的下载、解压、预处理，可以通过`--skip_xxx`跳过某个步骤。每个任务具有不同的准备工作步骤，详见各任务的`prepare.py`
```
python prepare.py --skip_download   // 跳过数据集下载
python prepare.py --skip_extract    // 跳过数据集解压
```
下载数据集和模型时可能需要代理，使用`--proxy`指定，如开启clash-verge后：
```
python prepare.py --proxy 127.0.0.1:7897
```
#### 3.2 运行基准测试
运行`run.py`
```
python run.py
```
指定`--batch_size`
```
python run.py --batch_size 8
```
# 4. 运行结果
每个任务的运行结果保存在各个目录下的`results/batch{args.batch_size}.csv`，包含运行时间和评价指标
```
- aibench
    ├──any_task                             // 某个任务
        ├──results
            ├──batch{args.batch_size}.csv   // 运行结果
    ...
```
如：
* image_classification_train, batch_size=16
```
train t (s),acc train t (s),mAP,CP,CR,CF1,OP,OR,OF1
169.56639528274536,169.56639528274536,0.9143797585012343,0.8833657999977961,0.8031335897948647,0.8413412425147263,0.8928304146085916,0.8505632396977043,0.8711844603475974
169.29734301567078,338.86373829841614,0.9334856433041161,0.916783531712492,0.8333048658747406,0.8730532457481419,0.9365325077399381,0.8626835876229859,0.8980924812588139
...
```
* image_classification_inference, batch_size=32
```
infer t (s),acc
81.98050737380981,0.76146
```