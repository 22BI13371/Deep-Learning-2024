import numpy as np
import pandas as pd

data_file = "data/xorData.csv"

w1 = np.array([[1.5, -0.5], 
              [-1, 1],
              [-1, 1]])

w2 = np.array([1.5, 1, 1])

data = pd.read_csv(data_file)

x_train = np.asarray(data.drop({"Result"}, axis=1))
y_train = np.asarray(data["Result"])

# print(x_train.shape)
# print(y_train.shape)

def signmoid(x):
  return 1 / (1 + np.exp(-x))

def feed_forward(x, w1, w2):
  z1 = x.dot(w1)
  a1 = signmoid(z1)
  
  a1 = np.insert(a1, 0, [1])
  z2 = a1.dot(w2)
  a2 = signmoid(z2)
  return float(a2)

output = []

for i in range(len(x_train)):
  output.append(round(feed_forward(x_train[i], w1, w2)))
  
print(output)