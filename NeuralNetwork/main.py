import pandas as pd
import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(in_features=21*3, out_features=100)
        self.first_hidden_layer = nn.Linear(in_features=100, out_features=100)
        self.second_hidden_layer = nn.Linear(in_features=100, out_features=100)
        self.output_layer = nn.Linear(in_features=100, out_features=24)

    def forward(self, x):
        x = f.relu(self.input_layer(x))
        x = f.relu(self.first_hidden_layer(x))
        x = f.relu(self.second_hidden_layer(x))
        x = f.sigmoid(self.output_layer(x))
        return x


def get_data(csv_data, ratio=0.8):
    csv_data = pd.read_csv("../Data_preprocessor/Processed_Data/combined_data.csv")
    # shuffle data
    csv_data = csv_data.sample(frac=1).reset_index(drop=True)

    # Each row has 21 points with x, y, z coordinates
    # We will flatten this data to 1D array
    # We will also normalize the data
    # The last column contains label, assign it to y
    x = csv_data.iloc[:, :-1].values
    y = csv_data.iloc[:, -1].values

    # Normalize data
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    x_train, x_test = x[:int(len(x) * ratio)], x[int(len(x) * ratio):]
    y_train, y_test = y[:int(len(y) * ratio)], y[int(len(y) * ratio):]

    return x_train, x_test, y_train, y_test


def train(model: Net, data):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def main():
    model = Net()
    x_train, x_test, y_train, y_test = get_data("../Data_preprocessor/Processed_Data/combined_data.csv")
    train(model, (x_train, y_train))


if __name__ == "__main__":
    main()
