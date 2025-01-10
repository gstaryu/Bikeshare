# -*- coding: UTF-8 -*-
"""
@Project: Bikeshare
@File   : models.py
@IDE    : PyCharm
@Author : staryu
@Date   : 2024/12/23 11:09
@Doc    : 训练代码
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import BikeDataset, MultiTaskBikeDataset


def train_model(model, train_loader, criterion, optimizer, num_epochs, device, model_type):
    """训练模型"""
    train_losses = []

    pbar = tqdm(range(num_epochs), desc='Training', ncols=100)
    for epoch in pbar:
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        pbar.set_postfix({'Loss': f'{train_loss / len(train_loader):.4f}'})

    return train_losses


def train_model_multi(model, train_loader, criterion, optimizer, num_epochs, device, model_type):
    """训练多任务模型"""
    train_losses = []

    pbar = tqdm(range(num_epochs), desc='Trainings', ncols=100)
    for epoch in pbar:
        model.train()
        train_loss = 0
        for batch_x, casual_y, registered_y, total_y in train_loader:
            batch_x = batch_x.to(device)
            casual_y = casual_y.to(device)
            registered_y = registered_y.to(device)
            total_y = total_y.to(device)

            optimizer.zero_grad()
            casual_pred, registered_pred, total_pred, _ = model(batch_x)

            casual_loss = criterion(casual_pred, casual_y)
            registered_loss = criterion(registered_pred, registered_y)
            total_loss = criterion(total_pred, total_y)

            loss = 0.3 * casual_loss + 0.3 * registered_loss + 0.4 * total_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()

        train_losses.append(train_loss / len(train_loader))

        pbar.set_postfix({'Loss': f'{train_loss / len(train_loader):.4f}'})

    return train_losses


def evaluate_model(model, test_loader, criterion, device, model_type='lstm'):
    """评估模型"""
    model.eval()
    test_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            test_loss += criterion(outputs, batch_y).item()

            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))

    return mse, mae, predictions, actuals


def evaluate_model_multi(model, test_loader, criterion, device, model_type='lstm'):
    """评估多任务模型"""
    model.eval()
    test_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, casual_y, registered_y, total_y in test_loader:
            batch_x = batch_x.to(device)
            casual_y = casual_y.to(device)
            registered_y = registered_y.to(device)
            total_y = total_y.to(device)

            _, _, outputs, _ = model(batch_x)

            test_loss += criterion(outputs, total_y)

            predictions.extend(outputs.cpu().numpy())
            actuals.extend(total_y.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))

    return mse, mae, predictions, actuals


def plot_results(predictions, actuals, title):
    """绘制结果图（只取第一阶段）"""
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual', color='b', alpha=0.6)
    plt.plot(predictions, label='Predicted', color='r', alpha=0.6)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Bike Rentals')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f'./output/{title}.png', dpi=300)
    plt.show()


def plot_losses(losses, title='Training Losses'):
    """损失函数下降曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(losses['lstm'], label='LSTM', color='b')
    plt.plot(losses['transformer'], label='Transformer', color='g')
    # plt.plot(losses['transformer multitask'], label='Transformer multitask', color='r')
    plt.plot(losses['tcn'], label='TCN', color='y')
    plt.plot(losses['tcn multitask'], label='TCN-M', color='r')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f'{title}.png', dpi=300)
    plt.show()


def run_experiment(model, train_data, test_data, output_window, model_type, batch_size=32,
                   learning_rate=0.001, num_epochs=100):
    """运行实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if model_type in ['transformer multitask', 'tcn multitask']:
        train_dataset = MultiTaskBikeDataset(train_data, output_window=output_window)
        test_dataset = MultiTaskBikeDataset(test_data, output_window=output_window)
    else:
        train_dataset = BikeDataset(train_data, output_window=output_window)
        test_dataset = BikeDataset(test_data, output_window=output_window)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if model_type in ['transformer multitask', 'tcn multitask']:
        train_losses = train_model_multi(
            model, train_loader, criterion, optimizer,
            num_epochs, device, model_type
        )

        mse, mae, predictions, actuals = evaluate_model_multi(
            model, test_loader, criterion, device, model_type
        )

    else:
        train_losses = train_model(
            model, train_loader, criterion, optimizer,
            num_epochs, device, model_type
        )

        mse, mae, predictions, actuals = evaluate_model(
            model, test_loader, criterion, device, model_type
        )

    print(f'MSE: {mse:.2f}, MAE: {mae:.2f}')

    return mse, mae, predictions, actuals, train_losses
