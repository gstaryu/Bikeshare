# Bikeshare
该项目为苏州大学2024级学硕的机器学习大作业。

任务：根据所提供的数据对未来单车租赁数量 cnt 进行预测。基于过去 I=96 小时的数据曲线来预测未来（ i O=96 小时（短期预测）和（ ii O=240 小时（长期预测）两种长度的变化曲线（分别训练）。

`dataset.py`：处理共享单车数据

`models.py`：包含LSTM、Transformer、TCN和TCN-M

`train.py`：训练模型的代码

`main.py`：训练入口

LSTM、Transformer、TCN是经典的机器学习架构，结合共享单车数据，在TCN的基础上提出了TCN结合多任务（TCN-M）的模型。

![model](https://github.com/user-attachments/assets/04853435-2366-4175-a32f-cbe89424a09e)

课程报告的Overleaf模板随本仓库提供。
