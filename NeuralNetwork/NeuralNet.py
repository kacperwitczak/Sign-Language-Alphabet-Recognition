from torch import nn
import torch.nn.functional as f


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
