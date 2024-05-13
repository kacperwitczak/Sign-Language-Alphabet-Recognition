import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import os
from Common.utils import load_model
from data import get_data
from NeuralNetwork.NeuralNet import Net
import json


def train(model: Net, data, epochs=10, batch_size=64, learn_rate=0.001, mom=0.9):
    x_train, y_train = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    x_train = torch.tensor(x_train.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.int64).to(device)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

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

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')


def test(model: Net, data):
    x_test, y_test = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test = torch.tensor(x_test.values, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)

    model.eval()
    with torch.inference_mode():
        pred = model(x_test)
        pred = torch.argmax(pred, dim=1)
        accuracy = (pred == y_test).float().mean()
        print(f'Accuracy: {accuracy}')


def save_model(model, path, n_in, n_out, hiddens):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    model_info = {
        'model_state_dict': model.state_dict(),
        'n_in': n_in,
        'n_out': n_out,
        'hiddens': hiddens
    }

    torch.save(model_info, path)


def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    model_path = config['model_path']
    epochs = config['epochs']
    batch_size = config['batch_size']
    learn_rate = config['learn_rate']
    momentum = config['momentum']
    hiddens = config['hiddens']
    load = config['load_model']

    x_train, x_test, y_train, y_test, label_count, feature_count = get_data()
    if load:
        model = load_model(Net, model_path)
        model.eval()
    else:
        model = Net(feature_count, hiddens, label_count)
        model.train()
        train(model, (x_train, y_train), epochs=epochs, learn_rate=learn_rate, mom=momentum, batch_size=batch_size)
        save_model(model, model_path, feature_count, label_count, hiddens)

    test(model, (x_test, y_test))


if __name__ == "__main__":
    main()