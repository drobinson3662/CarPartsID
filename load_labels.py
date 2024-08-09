import pandas as pd


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return df


if __name__ == '__main__':
    labels = load_labels('car parts.csv')
    print(labels.head())
