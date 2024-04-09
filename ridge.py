# ridge.py (original)
import numpy as np

class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.weights_ = np.linalg.solve(X.T @ X + self.alpha * np.eye(X.shape[1]), X.T @ y)

    def predict(self, X):
        return X @ self.weights_
#despues del ref

# ridge.py (refactored)
import numpy as np

class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _solve_normal_equation(self, K, b):
        return np.linalg.solve(K + self.alpha * np.eye(K.shape[0]), b)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.weights_ = self._solve_normal_equation(X.T @ X, X.T @ y)

    def predict(self, X):
        return X @ self.weights_

'''En el archivo refactorizado, se separa la inversión de la matriz en un método privado _resolver_ecuacion_normal, lo que hace que el código sea más modular y fácil de leer..'''