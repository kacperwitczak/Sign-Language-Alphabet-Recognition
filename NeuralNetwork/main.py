import pandas as pd
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np


class Net(nn.Module):
    def __init__(self, n_in, n_hiddens, n_out):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(n_i, n_o) for n_i, n_o in zip([n_in] + n_hiddens, n_hiddens + [n_out])]
        )

    def forward(self, x_data):
        x = x_data
        for layer in self.layers[:-1]:
            x = f.relu(layer(x))
        return f.softmax(self.layers[-1](x))


def get_data(ratio=0.8):
    csv_data = pd.read_csv("../Processed_Data/combined_data.csv")
    # shuffle data
    csv_data = csv_data.sample(frac=1).reset_index(drop=True)

    # Column labeled as 'label' is the target y
    y = csv_data['label']

    # each y is a char, so we need to convert it to a number
    y = y.apply(lambda char: ord(char) - ord('A'))

    label_count = len(np.unique(y))

    # The rest of the columns are the input x
    x = csv_data.drop(columns='label')
    feature_count = x.shape[1]

    x_train, x_test = x[:int(len(x) * ratio)], x[int(len(x) * ratio):]
    y_train, y_test = y[:int(len(y) * ratio)], y[int(len(y) * ratio):]

    return x_train, x_test, y_train, y_test, label_count, feature_count


def train(model: Net, data, epochs=10, batch_size=64):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            # Compute prediction and loss
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            # Backpropagation
            optimizer.zero_grad()
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


def main():
    x_train, x_test, y_train, y_test, label_count, feature_count = get_data()
    model = Net(feature_count, [32, 64, 32], label_count)
    train(model, (x_train, y_train), epochs=1000)
    test(model, (x_test, y_test))


if __name__ == "__main__":
    main()
