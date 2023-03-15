import numpy as np
from collections import Counter

# calculate Eudclidean distance
def calc_distance(x1, x2):
    dist = np.sqrt(np.sum((x1 - x2)**2))
    return dist


class KNN:
    def __init__(self, k=1):
        self.k = k

    # fit trainset into model
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # classify the test data
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # calculate the distances between
        dist = [calc_distance(x, x_train) for x_train in self.X_train]

        # get kNN
        k_nearest = np.argsort(dist)[:self.k]
        k_labels = [self.y_train[i] for i in k_nearest]

        # labeling test data
        label = Counter(k_labels).most_common()
        return label
