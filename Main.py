import torch
import LSTM
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def train_model(model, train_data, train_labels, test_data=None, test_labels=None):
    loss_fn = torch.nn.MSELoss(reduction='sum')

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 60

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        model.reset_hidden_state()

        y_pred = model(train_data)

        loss = loss_fn(y_pred.float(), train_labels)

        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(test_data)
                test_loss = loss_fn(y_test_pred.float(), test_labels)
            test_hist[t] = test_loss.item()

            if t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')

        train_hist[t] = loss.item()

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

    return model.eval(), train_hist, test_hist

if __name__ == "__main__" :
    sns.set(style='whitegrid', palette='muted', font_scale=1.2)

    HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

    sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

    rcParams['figure.figsize'] = 14, 10
    register_matplotlib_converters()

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    df = pd.read_csv('Data/time_series_19-covid-Confirmed.csv')
    print(df.head())

    df = df.iloc[:, 4:]
    df.isnull().sum().sum()
    daily_cases = df.sum(axis=0)
    daily_cases.index = pd.to_datetime(daily_cases.index)

    daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)

    seq_length = 5
    scaler = MinMaxScaler()

    test_data_size = 14

    train_data = daily_cases[:-test_data_size]
    test_data = daily_cases[-test_data_size:]


    scaler = scaler.fit(np.expand_dims(train_data, axis=1))

    train_data = scaler.transform(np.expand_dims(train_data, axis=1))

    test_data = scaler.transform(np.expand_dims(test_data, axis=1))


    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    model = LSTM.CoronaVirusPredictor(
        n_features=1,
        n_hidden=512,
        seq_len=seq_length,
        n_layers=2
    )

    model, train_hist, test_hist = train_model(model, X_train, y_train, X_test, y_test)

    with torch.no_grad():
      test_seq = X_test[:1]
      preds = []
      for _ in range(len(X_test)):
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()

    true_cases = scaler.inverse_transform(
        np.expand_dims(y_test.flatten().numpy(), axis=0)
    ).flatten()

    predicted_cases = scaler.inverse_transform(
      np.expand_dims(preds, axis=0)
    ).flatten()

    plt.plot(
      daily_cases.index[:len(train_data)],
      scaler.inverse_transform(train_data).flatten(),
      label='Historical Daily Cases'
    )

    plt.plot(
      daily_cases.index[len(train_data):len(train_data) + len(true_cases)],
      true_cases,
      label='Real Daily Cases'
    )

    plt.plot(
      daily_cases.index[len(train_data):len(train_data) + len(true_cases)],
      predicted_cases,
      label='Predicted Daily Cases'
    )

    plt.legend()
    plt.show()

# All Data
#     scaler = scaler.fit(np.expand_dims(daily_cases, axis=1))
#
#     all_data = scaler.transform(np.expand_dims(daily_cases, axis=1))
#
#     X_all, y_all = create_sequences(all_data, seq_length)
#
#     X_all = torch.from_numpy(X_all).float()
#     y_all = torch.from_numpy(y_all).float()
#
#     model = LSTM.CoronaVirusPredictor(
#         n_features=1,
#         n_hidden=512,
#         seq_len=seq_length,
#         n_layers=10
#     )
#     model, train_hist, _ = train_model(model, X_all, y_all)
#
#     DAYS_TO_PREDICT = 3
#
#     with torch.no_grad():
#         test_seq = X_all[:1]
#         preds = []
#         for _ in range(DAYS_TO_PREDICT):
#             y_test_pred = model(test_seq)
#             pred = torch.flatten(y_test_pred).item()
#             preds.append(pred)
#             new_seq = test_seq.numpy().flatten()
#             new_seq = np.append(new_seq, [pred])
#             new_seq = new_seq[1:]
#             test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
#
#     predicted_cases = scaler.inverse_transform(
#         np.expand_dims(preds, axis=0)
#     ).flatten()
#
#     predicted_index = pd.date_range(
#         start=daily_cases.index[-1],
#         periods=DAYS_TO_PREDICT + 1,
#         closed='right'
#     )
#
#     predicted_cases = pd.Series(
#         data=predicted_cases,
#         index=predicted_index
#     )
#
#     plt.plot(daily_cases, label='Historical Daily Cases')
#     plt.plot(predicted_cases, label='Predicted Daily Cases')
#     plt.legend();
#     plt.show()