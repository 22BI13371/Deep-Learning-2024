from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_file = "data/data2.csv"

data = pd.read_csv(data_file)

# print(data["Loan Decision"])

X_train = data.drop(["Loan Decision"], axis=1)
Y_train = data["Loan Decision"]

logreg = LogisticRegression(penalty='l2', tol=1e-8, solver='lbfgs', max_iter=500)
# logreg = LogisticRegression()

results = logreg.fit(X=X_train, y=Y_train)

# print(results.coef_)
# print(pd.DataFrame({"Feature":X_train.columns.tolist(),"Coefficients":results.coef_[0]}))
# print(results.intercept_)

# Retrieve the model parameters.
b = results.intercept_[0]
w1, w2 = results.coef_.T
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

plt.plot(data["Salary"], data["Working Time"], 'o')


# Plot the data and the classification with the decision boundary.
xmin, xmax = 4, 10
ymin, ymax = 0, 3
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

plt.show()




