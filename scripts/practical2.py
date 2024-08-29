import matplotlib.pyplot as plt
import math

x = [i for i in range(-100, 101)]
y = [math.pow(i, 2) for i in x]

# plt.xlabel("x")
# plt.ylabel("y")
# plt.plot(x, y)
# plt.show()

def gradientdDescent(x, error = 0.01, learningRate = 0.01, iterations = 100):
  xArray = []
  
  for i in range(0, iterations):
    x = x - learningRate * (2*x)
    xArray.append(x)
    
    if math.pow(x, 2) < error:
      break
    
  plt.plot(xArray)
  plt.show()  
  return x


print(gradientdDescent(-2, error=0.001,  iterations=5000 ))
  