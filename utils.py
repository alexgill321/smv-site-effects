import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

def read_data(train_path: str, test_path: str):
    """ Read data from train and test files

    Args:
        train_path: path to train file
        test_path: path to test file

    Returns:
        pandas dataframes for train and test data
    """

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train = train.drop(columns=["ID","subID","timepoint"])
    test = test.drop(columns=["ID","subID","timepoint"])

    return train, test

def generate_site_data_labels(data):
    label_list = []
    label_names = []
    for site in data["site"].unique():
        labels = data["site"] == site
        label_list.append(labels.astype(int).to_list())
        label_list.append(labels.astype(int).to_list())
        label_names.append(site)

    data.drop(columns=["site"], inplace=True)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(data)

    return train_data, label_list, label_names


class SiteDataset(Dataset):
    """ Site dataset
    
    Holds the data and labels for a single site. labels are 0 or 1 if the subject is from the site or not.

    Args:
        data: array of data of size (m x n)
        labels: array of labels of size (1 x m)

    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]