def KNN(train_set_name, test_set_name, knn_number):
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