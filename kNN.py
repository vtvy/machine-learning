import os
from collections import Counter

import numpy as np
import sys
np.set_printoptions(suppress=True,linewidth=np.nan)

# iris, optics, letter, faces, fp
train_set_name = input("Enter the file name of trainset:")
test_set_name = input("Enter the file name of testset:")
knn_number = input("Enter the number of nearest neighbors (k):")


def KNN(train_set_name, test_set_name, knn_number):
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
            if ("," in line):
                row = line.split(",")
            else:
                row = line.split(" ")
            # Convert strings to float values:
            for idx in range(len(row) - 1):
                row[idx] = float(row[idx])
            # convert class to int
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

    # get a list of classes in dataset by using Counter library
    sa = Counter([items[-1] for items in testset])
    class_list = list(Counter([items[-1] for items in dataset]).keys())
    class_number = len(class_list)

    startDistance = max(class_list) - class_number + 1
    if (startDistance > 0):
        matrix = [[0 for col in range(class_number + startDistance)] for row in range(class_number + startDistance)]
    else:
        matrix = [[0 for col in range(class_number)] for row in range(class_number)]
        
    # result_file = open(train_set_name + knn_number + ".txt" , "x")
    for example in testset:  # loop every example in testset
        dataset_number = len(dataset)
        distance_result = [[0 for col in range(2)] for row in range(dataset_number)]
        datapoint_number = len(dataset[0]) - 1
        for i in range(
            dataset_number
        ):  # for each example, loop every row of dataset to compute distance
            temp_distance = 0
            for j in range(datapoint_number):  # loop each data points
                temp_distance += abs(dataset[i][j] - example[j])
            distance_result[i][0] = temp_distance
            distance_result[i][1] = dataset[i][datapoint_number]  # store the class\
        distance_result.sort(key=lambda row: row[0])
        k_nearest_value = [
            items[1] for items in distance_result[0:knn_number]
        ]  # get list of the class from knn dataset
        # print(k_nearest_value)
        count = Counter(k_nearest_value)
        predict_class = [item[0] for item in count.most_common(1)]
        i_matrix = example[datapoint_number]
        j_matrix = predict_class[0]
        # print(str(i_matrix) + " predict " + str(j_matrix))
        matrix[i_matrix][j_matrix] += 1

    print("train file name: " + train_set_name)
    print("test file name: " + test_set_name)
    print("knn number: " + str(knn_number))

    for line in matrix:
        print(line)
    accuracy = (
        sum([matrix[i][i] for i in range(len(matrix))]) * 100 / sum(sum(matrix, []))
    )
    print("accuracy: " + str(accuracy))


KNN(train_set_name, test_set_name, int(knn_number))
