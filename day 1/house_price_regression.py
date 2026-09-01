import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([[500], [750], [1000], [1250], [1500]])  # sq ft
y = np.array([150000, 210000, 280000, 310000, 400000])  # price

model = LinearRegression()
model.fit(X, y)

print("Slope (price per sq ft):", model.coef_)
print("Intercept:", model.intercept_)

pred = model.predict([[900]])
print("Predicted price for 900 sqft:", pred)

plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.show()
