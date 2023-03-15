from csv import reader
from math import sqrt
import csv
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# get dataset from trainset and testset file
def get_data(filename, delimiter):
    data = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=delimiter)
        for row in csv_reader:
            if not row:
                continue
            data.append([float(i) for i in row])
    return data


# check if the file use space or comma for seperating data
def check_delimeter(filename):
    with open(filename, 'r') as file:
        if("," in file.read()):
            return 1
        else:
            return 0


def run(trainset, testset, k):
    n = check_delimeter(trainset)
    if n:
        train = get_data(trainset, ",")
        test = get_data(testset, ",")
    else:
        train = get_data(trainset, " ")
        test = get_data(testset, " ")
    
    X_train = np.array([x[:-1] for x in train])
    y_train = np.array([int(x[:-1]) for x in train])
    X_test = np.array([x[:-1] for x in test])
    y_test = np.array([int(x[:-1]) for x in test])



