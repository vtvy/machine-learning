import csv
from csv import reader
import numpy as np

def get_data(filename, delimiter):
    data = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=delimiter)
        for row in csv_reader:
            if not row:
                continue
            data.append([float(i) for i in row])
    return data

def run(filename, delimiter):
    data = get_data(filename, delimiter)
    X_train = np.array([int(x[-1:]) for x in data])
    return X_train

print(run("D:\kNN\data\iris\iris.trn", ","))