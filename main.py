import os
import pandas as pd
import utils as ut
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, balanced_accuracy_score


def main():
    """ Main Function: Independent SVC for each site for train and test data
    
    This function will read in the train and test data, generate the labels for each site, and then run a linear SVM
    on each site. The results are printed to the console.
    
    """
    train_df, test_df = ut.read_data(os.getcwd() + '/data/TrainCons.csv', os.getcwd() + '/data/TestCons.csv')
    train_data, train_label_list, train_label_names = ut.generate_site_data_labels(train_df)
    test_data, test_label_list, test_label_names = ut.generate_site_data_labels(test_df)

    balanced_accuracy = make_scorer(balanced_accuracy_score)
    
    train_res_dict = {}
    for train_labels, label_name in zip(train_label_list, train_label_names):
        print("Training set number of samples for {}: {}".format(label_name, sum(train_labels)))
        scores = cross_val_score(SVC(kernel='linear', class_weight='balanced'), 
                                 train_data, 
                                 train_labels, 
                                 cv=KFold(shuffle=True, n_splits=2), 
                                 scoring=balanced_accuracy
                                 )
        train_res_dict[label_name] = scores.mean()

    test_res_dict = {}
    for test_labels, label_name in zip(test_label_list, test_label_names):
        print("Test set number of samples for {}: {}".format(label_name, sum(test_labels)))
        scores = cross_val_score(SVC(kernel='linear', class_weight='balanced'), 
                                 test_data, 
                                 test_labels, 
                                 cv=KFold(shuffle=True, n_splits=2), 
                                 scoring=balanced_accuracy
                                 )
        test_res_dict[label_name] = scores.mean()

    print("Train Results:")
    print(train_res_dict)
    print("Test Results:")
    print(test_res_dict)


if __name__ == '__main__':
    main()