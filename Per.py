import os
import numpy as np
import math

# set the terminal to be max width for printing a row in only 1 line.
np.set_printoptions(suppress=True, linewidth=np.nan)


# function to read file and append to a 2D list and return this list
def read_data(data_file):
    data_list = []
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
        # Append to a 2D list:
        data_list.append(row)
    return data_list


class Per:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.max_iterations = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iterations):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                predicted_output = np.sign(linear_output)

                if predicted_output != y[i]:
                    update = self.learning_rate * (y[i] - predicted_output)
                    self.weights += update * X[i]
                    self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        if linear_output > 0:
            return 1
        return -1


def Main():
    # spam, ovarian, leukemia
    train_set_name = input("Enter the file name of trainset:")
    test_set_name = input("Enter the file name of testset:")
    learning_rate = input("Enter the value of learning rate:")
    epochs = input("Enter the number of epochs:")

    dataset = []
    testset = []
    if train_set_name == "leukemia":
        train_set_files = [
            f for f in os.listdir("data/" + train_set_name) if f.endswith(".trn")
        ]
        train_set_file = open("data/" + train_set_name + "/" + train_set_files[0], "r")

        test_set_files = [
            f for f in os.listdir("data/" + test_set_name) if f.endswith(".tst")
        ]
        test_set_file = open("data/" + test_set_name + "/" + test_set_files[0], "r")
        # declare and call read_data function
        dataset = read_data(train_set_file)
        testset = read_data(test_set_file)
        train_set_file.close()
        test_set_file.close()
    else:
        example_file = open(
            "data/" + train_set_name + "/" + train_set_name + ".data", "r"
        )
        example = read_data(example_file)
        number_of_datapoint = len(example)
        number_of_trainpoint = math.ceil(2 * number_of_datapoint / 3)
        dataset = example[:number_of_trainpoint][:]
        testset = example[number_of_trainpoint:][:]
        example_file.close()

    X_train = np.array([row[:-1] for row in dataset])
    y_train = np.array([row[-1] for row in dataset])
    X_test = [row[:-1] for row in testset]
    y_test = [row[-1] for row in testset]

    # declare and train the model
    perception = Per(float(learning_rate), int(epochs))
    perception.fit(X_train, y_train)

    matrix = [[0 for row in range(2)] for row in range(2)]
    # predict and print confusion matrix
    for i in range(len(y_test)):  # loop every example in testset
        predict_class = perception.predict(X_test[i])
        i_matrix = 0
        j_matrix = 0
        if y_test[i] == 1:
            i_matrix = 1
        if predict_class == 1:
            j_matrix = 1
        matrix[i_matrix][j_matrix] += 1
    print("train file name: " + train_set_name)
    print("test file name: " + test_set_name)
    print("learning rate: " + learning_rate)
    print("maximum number of epochs maxit: " + epochs)
    for line in matrix:
        print(line)
    accuracy = (
        sum([matrix[i][i] for i in range(len(matrix))]) * 100 / sum(sum(matrix, []))
    )
    print("accuracy: " + str(accuracy))


Main()
