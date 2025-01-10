# -*- coding: UTF-8 -*-
"""
@Project: Bikeshare
@File   : test.py
@IDE    : PyCharm
@Author : staryu
@Date   : 2025/1/9 17:57
@Doc    : 
"""
from train import plot_losses
import json

data = json.load(open('../data/100epo_long.json', 'r'))

# 删除transformer multitask
data.pop('transformer multitask')


plot_losses(data, title='Long-term Training Loss')