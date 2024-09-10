from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_file = "data/xorData.csv"

data = pd.read_csv(data_file)

X_train = data.drop(["Result"], axis=1)
Y_train = data["Result"]

neural_net = MLPClassifier(hidden_layer_sizes=(2,), activation="logistic", tol=1e-10, max_iter=10000000)

neural_net.fit(X=X_train, y=Y_train)

print(neural_net.predict(X_train))