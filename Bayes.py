import os
from sklearn.naive_bayes import GaussianNB
import numpy as np

np.set_printoptions(suppress=True, linewidth=np.nan)

train_set_name = input("Enter the file name of trainset:")
test_set_name = input("Enter the file name of testset:")


def NB(train_set_name, test_set_name):
    train_set_files = [
        f for f in os.listdir("data/" + train_set_name) if f.endswith(".trn")
    ]
    train_set_file = open("data/" + train_set_name + "/" + train_set_files[0], "r")

    test_set_files = [
        f for f in os.listdir("data/" + test_set_name) if f.endswith(".tst")
    ]
    test_set_file = open("data/" + test_set_name + "/" + test_set_files[0], "r")

    # function to read file and append to a 2D list
    def read_data(data_list, data_file):
        for line in data_file:
            if "," in line:
                row = line.split(",")
            else:
                row = line.split(" ")
            # Convert strings to float values:
            for idx in range(len(row) - 1):
                row[idx] = float(row[idx])
            # convert class to int
            row[-1] = int(row[-1])
            # convert class to int value
            row[-1] = int(row[-1])
            # Append to a 2D list:
            data_list.append(row)

    # declare and call read_data function
    dataset = []
    read_data(dataset, train_set_file)
    testset = []
    read_data(testset, test_set_file)
    train_set_file.close()
    test_set_file.close()

    gaussianNB = GaussianNB()

    X_train = [row[:-1] for row in dataset]
    y_train = [row[-1] for row in dataset]
    X_test = [row[:-1] for row in testset]
    y_test = [row[-1] for row in testset]
    gaussianNB.fit(X_train, y_train)

    y_predict = gaussianNB.predict(X_test)

    from collections import Counter

    class_list = list(Counter(y_test).keys())
    class_number = len(class_list)
    # declare a 2D list with 0 value default to store the confusion matrix
    startDistance = max(class_list) - class_number + 1
    if startDistance > 0:
        matrix = [
            [0 for col in range(class_number + startDistance)]
            for row in range(class_number + startDistance)
        ]
    else:
        matrix = [[0 for col in range(class_number)] for row in range(class_number)]

    for idx in range(len(y_test)):
        i_matrix = y_test[idx]
        j_matrix = y_predict[idx]
        matrix[i_matrix][j_matrix] += 1

    print("train file name: " + train_set_name)
    print("test file name: " + test_set_name)

    from sklearn.metrics import accuracy_score, confusion_matrix

    print("confusion matrix:")
    print(confusion_matrix(y_test, y_predict))

    print("accuracy (%): ", accuracy_score(y_test, y_predict) * 100)


NB(train_set_name, test_set_name)
