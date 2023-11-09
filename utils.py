import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    """ Generate labels for each site

    Args:
        data: pandas dataframe

    Returns:
        train_data: numpy array of train data
        label_list: list of labels for each site
        label_names: list of site names corresponding to label_list
    """
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