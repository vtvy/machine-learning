import matplotlib.pyplot as plt


# draw 1 point of class -1
plt.scatter([0], [0])

# draw 3 points of class +1
plt.scatter([0, 1, 1], [1, 0, 1])


# add 2 supporting planes
plt.plot(
    [-0.5, 1.5], [1.5, -0.5], label="Supporting plane", linestyle="--", color="orange"
)
plt.plot([-0.5, 0.5], [0.5, -0.5], label="Supporting plane", linestyle="--", color="c")

# add Support Vector
plt.plot([-0.5, 1], [1, -0.5], label="Support Vector model", color="r")
# set the label in location upper right
plt.legend(loc="upper right")

# set the limit of 2 dimensions
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

# set the label for each dimension
plt.xlabel("x2")
plt.ylabel("x1")

plt.show()
