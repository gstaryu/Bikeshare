# -*- coding: UTF-8 -*-
"""
@Project: Bikeshare
@File   : train.py
@IDE    : PyCharm
@Author : staryu
@Date   : 2024/12/23 11:10
@Doc    : 用于处理共享单车数据集的类
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


# from sklearn.preprocessing import MinMaxScaler


class BikeDataset(Dataset):
    def __init__(self, data, input_window=96, output_window=96, stride=1):
        self.data = torch.FloatTensor(data.values)
        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride  # 滑动窗口步长
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        for i in range(0, len(self.data) - self.input_window - self.output_window + 1, self.stride):
            input_seq = self.data[i:i + self.input_window]  # 输入包括所有特征以及目标值
            target_seq = self.data[i + self.input_window:i + self.input_window + self.output_window,
                         -1]  # 只包括目标值
            samples.append((input_seq, target_seq))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MultiTaskBikeDataset(Dataset):
    def __init__(self, data, input_window=96, output_window=96, stride=1):
        """
        专为多任务学习设计的数据集类
        """
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []

        # 分别获取 casual, registered, cnt 列
        casual_idx = self.data.columns.get_loc('casual')
        registered_idx = self.data.columns.get_loc('registered')
        cnt_idx = self.data.columns.get_loc('cnt')

        data_array = torch.FloatTensor(self.data.values)

        for i in range(0, len(self.data) - self.input_window - self.output_window + 1, self.stride):
            input_seq = data_array[i:i + self.input_window]

            # 为不同的目标值创建不同的目标序列
            casual_target = data_array[i + self.input_window:i + self.input_window + self.output_window, casual_idx]
            registered_target = data_array[i + self.input_window:i + self.input_window + self.output_window,
                                registered_idx]
            total_target = data_array[i + self.input_window:i + self.input_window + self.output_window, cnt_idx]

            samples.append((input_seq, casual_target, registered_target, total_target))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def prepare_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 去除空值
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # 去除不需要的列
    # drop_cols = ['instant', 'dteday', 'casual', 'registered']
    drop_cols = ['instant', 'dteday']
    train_df = train_df.drop(drop_cols, axis=1)
    test_df = test_df.drop(drop_cols, axis=1)

    return train_df, test_df
