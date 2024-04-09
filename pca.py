#antes del ref

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=1e-7, iterated_power='auto',
                 random_state=None):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        
#despues del ref

class PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=1e-7, power_iteration_number='auto',
                 random_state=None):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.power_iteration_number = power_iteration_number
        self.random_state = random_state


'''Razón para refactoring:

Renombré el parámetro iterated_power a power_iteration_number para reflejar mejor su significado.'''