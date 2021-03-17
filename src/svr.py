from sklearn.svm import SVR
import numpy as np


# class OSVR():
#     def __init__(self):
#         self.svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

#     def predict(self, X):
#         return self.svr.predict(X)
if __name__ == "__main__":
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
