# -*- coding: UTF-8 -*-
"""
@Project: Bikeshare
@File   : models.py
@IDE    : PyCharm
@Author : staryu
@Date   : 2024/12/23 11:09
@Doc    : 模型
"""
import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        # (batch_size, seq_length, input_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lstm_output, _ = self.lstm(x, (h0, c0))

        output = lstm_output[:, -1, :]  # 只取最后一个时间步的输出

        output = self.fc(output)
        return output


class PositionalEncoding(nn.Module):
    """
    Reference: https://github.com/hkproj/pytorch-transformer
    """

    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers,
                 dim_feedforward, output_size, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.input = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.output = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_size)
        )

    def forward(self, x):
        x = self.input(x)

        x = self.pos_encoder(x)

        transformer_output = self.transformer_encoder(x)

        output = transformer_output[:, -1, :]

        output = self.output(output)

        return output


class MultiTaskTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers,
                 dim_feedforward, output_size, dropout=0.1):
        super(MultiTaskTransformer, self).__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # casual
        self.casual = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_size)
        )

        # registered
        self.registered = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_size)
        )

        self.attention = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # (batch_size, seq_length, input_size)

        x = self.input_proj(x)

        x = self.pos_encoder(x)

        encoded = self.transformer_encoder(x)

        features = encoded[:, -1, :]

        casual_pred = self.casual(features)
        registered_pred = self.registered(features)

        attention_weights = self.attention(features)

        total_pred = casual_pred + registered_pred

        return casual_pred, registered_pred, total_pred, attention_weights


# Reference: https://github.com/locuslab/TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """
    TCN的基本模块，包含两层因果卷积和残差连接
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        初始化权重
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    TCN网络主体，包含多个TCN基本模块
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    """
    完整的TCN模型，包含TCN网络和输出层
    """

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.cnt = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, output_size)
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.cnt:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 转换输入维度顺序以适应TCN
        x = x.transpose(1, 2)
        tcn_output = self.tcn(x)
        # 只使用最后一个时间步的输出进行预测
        output = self.cnt(tcn_output[:, :, -1])
        return output


class MultitaskTCN(nn.Module):
    """
    多任务TCN模型，同时预测casual和registered用户数
    """

    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(MultitaskTCN, self).__init__()

        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # 共享特征提取后的任务特定层
        self.casual = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, output_size)
        )

        self.registered = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, output_size)
        )

    def init_weights(self):
        for head in [self.casual_head, self.registered_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # (batch_size, seq_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_length)

        tcn_output = self.tcn(x)

        features = tcn_output[:, :, -1]

        # 分别预测casual和registered
        casual_pred = self.casual(features)
        registered_pred = self.registered(features)

        # 合并预测结果
        total_pred = casual_pred + registered_pred

        return casual_pred, registered_pred, total_pred, None
