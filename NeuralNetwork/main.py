import pandas as pd
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset
import torch
from datetime import datetime
import os
import numpy as np

class Net(nn.Module):
    def __init__(self, n_in, n_hiddens, n_out):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(n_i, n_o) for n_i, n_o in zip([n_in] + n_hiddens, n_hiddens + [n_out])]
        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x_data):
        x = x_data
        for layer in self.layers[:-1]:
            x = f.relu(layer(x))
        return self.layers[-1](x)


def get_data(ratio=0.8):
    csv_data = pd.read_csv("../Processed_Data/combined_data.csv")
    csv_data = csv_data.sample(frac=1).reset_index(drop=True)

    y = csv_data['label']
    y = y.apply(lambda char: ord(char) - ord('A'))

    label_count = np.max(y) + 1

    x = csv_data.drop(columns='label')
    feature_count = x.shape[1]

    x_train, x_test = x[:int(len(x) * ratio)], x[int(len(x) * ratio):]
    y_train, y_test = y[:int(len(y) * ratio)], y[int(len(y) * ratio):]

    return x_train, x_test, y_train, y_test, label_count, feature_count


def train(model: Net, data, epochs=10, batch_size=64, learn_rate=0.001, mom=0.9):
    # Unpack data
    x_train, y_train = data

    # Convert pandas dataframes to torch tensors
    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.int64)

    # Create a DataLoader to handle batching
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss function - cross entropy
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=mom)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print loss every epoch
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')


def test(model: Net, data):
    x_test, y_test = data

    x_test = torch.tensor(x_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    model.eval()
    with torch.inference_mode():
        pred = model(x_test)
        pred = torch.argmax(pred, dim=1)
        accuracy = (pred == y_test).float().mean()
        print(f'Accuracy: {accuracy}')


def save_model(model, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(model.state_dict(), path)

def load_model(model_class, path, n_in, n_hiddens, n_out):
    model = model_class(n_in, n_hiddens, n_out)
    model.load_state_dict(torch.load(path))

    return model


def main():
    x_train, x_test, y_train, y_test, label_count, feature_count = get_data()
    hiddens = [64, 32]
    load = False
    if load:
        model = load_model(Net, "../Models/model_xd.pth", feature_count, hiddens, label_count)
        model.eval()  # Ustawienie modelu w tryb ewaluacji

    else:
        model = Net(feature_count, hiddens, label_count)
        model.train()  # Ustawienie modelu w tryb treningu
        train(model, (x_train, y_train), epochs=200, learn_rate=0.001, mom=0.9)
        save_model(model, "../Models/model_xd.pth")


    # W każdym przypadku testuj model w trybie ewaluacji
    test(model, (x_test, y_test))


if __name__ == "__main__":
    main()
