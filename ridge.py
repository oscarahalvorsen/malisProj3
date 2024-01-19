import numpy as np


class RidgeRegression():
    def __init__(self, learning_rate=0.01, iterations=1000, l2_penalty=1):
       self.learning_rate = learning_rate
       self.iterations = iterations
       self.l2_penalty = l2_penalty

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        for _ in range(self.iterations):
            self.update_weights()
