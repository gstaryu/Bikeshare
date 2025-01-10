# Bikeshare
该项目为苏州大学2024级学硕的机器学习大作业。

任务：根据所提供的数据对未来单车租赁数量 cnt 进行预测。基于过去 I=96 小时的数据曲线来预测未来（ i O=96 小时（短期预测）和（ ii O=240 小时（长期预测）两种长度的变化曲线（分别训练）。

`dataset.py`：处理共享单车数据

`models.py`：包含LSTM、Transformer、TCN和TCN-M

`train.py`：训练模型的代码

`main.py`：训练入口
