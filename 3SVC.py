import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# declare dataset
dataset = [
    [0.204000, 0.834000, 0],
    [0.222000, 0.730000, 0],
    [0.298000, 0.822000, 0],
    [0.450000, 0.842000, 0],
    [0.412000, 0.732000, 0],
    [0.298000, 0.640000, 0],
    [0.588000, 0.298000, 0],
    [0.554000, 0.398000, 0],
    [0.670000, 0.466000, 0],
    [0.834000, 0.426000, 0],
    [0.724000, 0.368000, 0],
    [0.790000, 0.262000, 0],
    [0.824000, 0.338000, 0],
    [0.136000, 0.260000, 1],
    [0.146000, 0.374000, 1],
    [0.258000, 0.422000, 1],
    [0.292000, 0.282000, 1],
    [0.478000, 0.568000, 1],
    [0.654000, 0.776000, 1],
    [0.786000, 0.758000, 1],
    [0.690000, 0.628000, 1],
    [0.736000, 0.786000, 1],
    [0.574000, 0.742000, 1],
]
X_train = np.array([row[:-1] for row in dataset])
y_train = np.array([row[-1] for row in dataset])

# declare 2 random arrays with 1000 elements
x = np.linspace(-0.5, 1.5, 1000)
y = np.linspace(-0.5, 1.5, 1000)

yy, xx = np.meshgrid(x, y)
xy = np.vstack([xx.ravel(), yy.ravel()]).T
fig = plt.gcf()
fig.set_size_inches(10, 10)

# fit SVC model
clf = SVC(kernel="rbf", C=200)
clf.fit(X_train, y_train)

# plot the decision function on the stack for each datapoint
dec_stack = clf.decision_function(xy).reshape(xx.shape)

# split data train into 2 arrays X1 and X2
X1 = [x[0] for x in X_train]
X2 = [x[1] for x in X_train]

plt.contour(xx, yy, dec_stack, levels=[0], linewidths=3, linestyles=["--", "dotted"])
plt.scatter(X1, X2, s=30, c=y_train, cmap="cool")
plt.show()
