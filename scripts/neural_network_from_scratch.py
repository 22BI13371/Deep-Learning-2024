import numpy as np
import pandas as pd

data_file = "data/xorData.csv"

w1 = np.array([[-1, 1],
              [-1, 1]])

w2 = np.array([1, 1])

w1_bias = np.array([1.5, -0.5])
w2_bias = -1.5

data = pd.read_csv(data_file)

x_train = np.asarray(data.drop({"Result"}, axis=1))
y_train = np.asarray(data["Result"])

# print(x_train.shape)
# print(y_train.shape)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def feed_forward(x, w1, w2, w1_bias, w2_bias):
  z1 = x.dot(w1) + w1_bias
  a1 = np.round(sigmoid(z1))
  
  z2 = a1.dot(w2) + w2_bias
  a2 = np.round(sigmoid(z2))
  return round(a2)

output = []

for i in range(len(x_train)):
  output.append(feed_forward(x_train[i], w1, w2, w1_bias, w2_bias))
print(output)