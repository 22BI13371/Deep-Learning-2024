import pandas as pd
import matplotlib.pyplot as plt

data_file = "data/dataPoints.csv"

data = pd.read_csv(data_file)

plt.xlabel("Square")
plt.ylabel("Price")
plt.title("Square Price correlation")

plt.plot(data["Square"], data["Price"], 'o')

plt.show()
