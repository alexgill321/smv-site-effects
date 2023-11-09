import torch
import torch.nn as nn


class LSVM(nn.Module):
    """ Linear Support Vector Machine

    This SVM will act as a binary classifier. It will take in a vector of features and output a single value
    between 0 and 1.

    Args:
        n_features: number of features in the data
    """
    def __init__(self, n_features):
        super(LSVM, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)