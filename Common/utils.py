import torch


def load_model(model_class, path):
    model_info = torch.load(path)

    model = model_class(model_info['n_in'], model_info['hiddens'], model_info['n_out'])
    model.load_state_dict(model_info['model_state_dict'])

    return model