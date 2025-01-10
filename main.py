# -*- coding: UTF-8 -*-
"""
@Project: Bikeshare
@File   : main.py
@IDE    : PyCharm
@Author : staryu
@Date   : 2024/12/236 19:24
@Doc    : 训练入口
"""
from dataset import prepare_data
from models import LSTMModel, TransformerModel, TCNModel, MultiTaskTransformer, MultitaskTCN
from train import run_experiment, plot_results, plot_losses
import numpy as np
from datetime import datetime
import json
import torch

import warnings

warnings.filterwarnings("ignore")

# 全局变量，用于保存训练过程中的损失值
short_term_losses = {
    'lstm': [],
    'transformer': [],
    'transformer_multitask': [],
    'tcn': [],
    'tcn_multitask': []
}

long_term_losses = {
    'lstm': [],
    'transformer': [],
    'transformer_multitask': [],
    'tcn': [],
    'tcn_multitask': []
}


def run_multiple_experiments(model_class, model_params, train_data, test_data,
                             model_type, num_runs=5):
    short_term_results = {
        'mse': [], 'mae': [],
        'predictions': [], 'actuals': [],
        'train_losses': []
    }
    long_term_results = {
        'mse': [], 'mae': [],
        'predictions': [], 'actuals': [],
        'train_losses': []
    }

    if model_type not in ['transformer multitask', 'tcn multitask']:
        train_data = train_data.drop(['casual', 'registered'], axis=1)
        test_data = test_data.drop(['casual', 'registered'], axis=1)

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")

        # Short-term
        print("Training Short-term Model...")
        model_params['output_size'] = 96
        model_short = model_class(**model_params)
        mse_short, mae_short, pred_short, act_short, losses_short = run_experiment(
            model=model_short,
            train_data=train_data,
            test_data=test_data,
            output_window=96,
            model_type=model_type
        )

        # Long-term
        print("Training Long-term Model...")
        model_params['output_size'] = 240
        model_long = model_class(**model_params)
        mse_long, mae_long, pred_long, act_long, losses_long = run_experiment(
            model=model_long,
            train_data=train_data,
            test_data=test_data,
            output_window=240,
            model_type=model_type
        )

        short_term_results['mse'].append(mse_short)
        short_term_results['mae'].append(mae_short)
        short_term_results['predictions'].append(pred_short)
        short_term_results['actuals'].append(act_short)

        long_term_results['mse'].append(mse_long)
        long_term_results['mae'].append(mae_long)
        long_term_results['predictions'].append(pred_long)
        long_term_results['actuals'].append(act_long)

    # 最终整合的结果
    final_results = {
        'short_term': {
            'mse_mean': np.mean(short_term_results['mse']),
            'mse_std': np.std(short_term_results['mse']),
            'mae_mean': np.mean(short_term_results['mae']),
            'mae_std': np.std(short_term_results['mae'])
        },
        'long_term': {
            'mse_mean': np.mean(long_term_results['mse']),
            'mse_std': np.std(long_term_results['mse']),
            'mae_mean': np.mean(long_term_results['mae']),
            'mae_std': np.std(long_term_results['mae'])
        }
    }

    # 分别绘制短期和长期最后一次预测的结果（也就是前96小时和前240小时）
    plot_results(pred_short[0], act_short[0], f"{model_type.capitalize()} Short-term Prediction (96 hours) Step 1")

    plot_results(pred_long[0], act_long[0], f"{model_type.capitalize()} Long-term Prediction (240 hours) Step 1")

    # 分别绘制短期和长期最后一次所有预测的平均结果
    plot_results(np.mean(pred_short, axis=1), np.mean(act_short, axis=1),
                 f"{model_type.capitalize()} Short-term Prediction (96 hours) Average")

    plot_results(np.mean(pred_long, axis=1), np.mean(act_long, axis=1),
                 f"{model_type.capitalize()} Long-term Prediction (240 hours) Average")

    short_term_losses[model_type] = losses_short
    long_term_losses[model_type] = losses_long

    # plot_results(
    #     np.mean([pred[0] for pred in short_term_results['predictions']], axis=1),
    #     short_term_results['actuals'][0],
    #     f"{model_type.upper()} Short-term Prediction (96 hours)"
    # )
    #
    # plot_results(
    #     np.mean([pred[0] for pred in long_term_results['predictions']], axis=0),
    #     long_term_results['actuals'][0],
    #     f"{model_type.upper()} Long-term Prediction (240 hours)"
    # )

    return final_results


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Load data...")
    train_data, test_data = prepare_data('./data/train_data.csv', './data/test_data.csv')
    input_size = train_data.shape[1]

    # 模型参数
    lstm_params = {
        'input_size': input_size - 2,
        'hidden_size': 128,
        'num_layers': 2,
        'output_size': None,
        'dropout': 0.2
    }

    transformer_params = {
        'input_size': input_size - 2,
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 3,
        'dim_feedforward': 256,
        'output_size': None,
        'dropout': 0.1
    }

    # transformer_multitask_params = {
    #     'input_size': input_size,
    #     'd_model': 128,
    #     'nhead': 8,
    #     'num_encoder_layers': 3,
    #     'dim_feedforward': 256,
    #     'output_size': None,
    #     'dropout': 0.1
    # }

    tcn_params = {
        'input_size': input_size - 2,
        'output_size': None,
        'num_channels': [64, 128, 256],
        'kernel_size': 3,
        'dropout': 0.2
    }

    tcn_multitask_params = {
        'input_size': input_size,
        'output_size': None,
        'num_channels': [64, 128, 256],
        'kernel_size': 3,
        'dropout': 0.2
    }

    # Run experiments
    print("\nRunning LSTM experiments...")
    lstm_results = run_multiple_experiments(
        model_class=LSTMModel,
        model_params=lstm_params,
        train_data=train_data,
        test_data=test_data,
        model_type='lstm'
    )

    print("\nRunning Transformer experiments...")
    transformer_results = run_multiple_experiments(
        model_class=TransformerModel,
        model_params=transformer_params,
        train_data=train_data,
        test_data=test_data,
        model_type='transformer'
    )

    # print("\nRunning Transformer Multitask experiments...")
    # transformer_multitask_results = run_multiple_experiments(
    #     model_class=MultiTaskTransformer,
    #     model_params=transformer_multitask_params,
    #     train_data=train_data,
    #     test_data=test_data,
    #     model_type='transformer multitask'
    # )

    print("\nRunning TCN experiments...")
    tcn_results = run_multiple_experiments(
        model_class=TCNModel,
        model_params=tcn_params,
        train_data=train_data,
        test_data=test_data,
        model_type='tcn'
    )

    print("\nRunning TCN Multitask experiments...")
    tcn_multitask_results = run_multiple_experiments(
        model_class=MultitaskTCN,
        model_params=tcn_multitask_params,
        train_data=train_data,
        test_data=test_data,
        model_type='tcn multitask'
    )

    results = {
        'lstm': lstm_results,
        'transformer': transformer_results,
        # 'transformer multitask': transformer_multitask_results,
        'tcn': tcn_results,
        'tcn multitask': tcn_multitask_results
    }

    plot_losses(short_term_losses, title='Short-term Training Losses')
    plot_losses(long_term_losses, title='Long-term Training Losses')

    def custom_encoder(obj):
        """解决报错，将 numpy.float32 转换为标准 float"""
        if isinstance(obj, np.float32):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=4, default=custom_encoder)
    with open(f"short_term_losses_{timestamp}.json", 'w') as f:
        json.dump(short_term_losses, f, indent=4, default=custom_encoder)
    with open(f"long_term_losses_{timestamp}.json", 'w') as f:
        json.dump(long_term_losses, f, indent=4, default=custom_encoder)


if __name__ == "__main__":
    main()
