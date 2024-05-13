import numpy as np
import pandas as pd


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
