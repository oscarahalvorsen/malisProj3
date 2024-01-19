from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#Ridge Regression implementation
class RidgeRegression():
    def __init__(self, learning_rate=0.01):
       self.learning_rate = learning_rate

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        
        self.w = np.identity(X.shape[1])
        self.w[0,0]=0
        
        self.w = np.linalg.inv(X.T.dot(X) + self.learning_rate*self.w ).dot(X.T).dot(Y)
    
    def predict(self, X):
        return X.dot(self.w)
    
    def mse(self, Y_true, Y_pred):
        return ((Y_true - Y_pred) ** 2).mean()
    
    def r_squared(self, Y_true, Y_pred):
        correlation_matrix = np.corrcoef(Y_true, Y_pred)
        correlation_xy = correlation_matrix[0,1]
        return correlation_xy**2
    
df = pd.read_csv('olympics_100m.csv')

# Prepare the data
X = df[['Year']].values
Y = df['Time'].values

# Add a column of ones to X to represent the bias term (intercept)
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Create and fit the model
model = RidgeRegression(learning_rate=0.01)
model.fit(X, Y)

# Make predictions (using the model for something would be the next step)
predictions = model.predict(X)

# Calculate and print MSE and R-squared
mse_value = model.mse(Y, predictions)
r_squared_value = model.r_squared(Y, predictions)
print(f"MSE: {mse_value}")
print(f"R-squared: {r_squared_value}")

# Plot the model alongside the data
plt.scatter(X[:, 1], Y, color='blue', label='Actual data')
plt.plot(X[:, 1], predictions, color='red', label='Ridge Regression Model')
plt.xlabel('Year')
plt.ylabel('Time')
plt.title('Olympic 100m Winning Times Over the Years')
plt.legend()
plt.show()
