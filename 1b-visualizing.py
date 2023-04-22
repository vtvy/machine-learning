import matplotlib.pyplot as plt

# declare 2 points A and B
a1, a2 = -1, 5 / 9
b1, b2 = 5 / 2, -1

# add 2 points in the classification model
plt.plot([a1, a2], [b1, b2], label="classification model", linewidth=1)
# set the label in location upper right
plt.legend(loc="upper right")

# draw 1 point of class -1
plt.scatter([0], [0])

# draw 3 points of class +1
plt.scatter([0, 1, 1], [1, 0, 1])

# set the limit of 2 dimensions
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

# set the label for each dimension
plt.xlabel("x2")
plt.ylabel("x1")

plt.show()
