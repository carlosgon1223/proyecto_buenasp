#antes del ref

def pairwise_distances(X, Y=None, metric='euclidean', n_jobs=None, **kwds):
    if Y is None:
        Y = X
    if isinstance(X, (KDTree, BallTree)):
        return _pairwise_distances_tree(X, Y, metric, n_jobs, **kwds)
    elif isinstance(Y, (KDTree, BallTree)):
        return _pairwise_distances_tree(Y, X, metric, n_jobs, **kwds)
    elif metric == 'precomputed':
        return _validate_precomputed(X, Y)
    elif issparse(X) and issparse(Y):
        return _pairwise_distances_csr(X, Y, metric, n_jobs, **kwds)
    elif issparse(X):
        return _pairwise_distances_csr(X, Y, metric, n_jobs, **kwds)
    elif issparse(Y):
        return _pairwise_distances_csr(Y, X, metric, n_jobs, **kwds)
    else:
        return _pairwise_distances_numpy(X, Y, metric, n_jobs, **kwds)
    
    #despues del ref
    
    def pairwise_distances(X, Y=None, metric='euclidean', n_jobs=None, **kwds):
        if Y is None:
        Y = X
    if isinstance(X, (KDTree, BallTree)) or isinstance(Y, (KDTree, BallTree)):
        return _pairwise_distances_tree(X, Y, metric, n_jobs, **kwds)
    if issparse(X) and issparse(Y):
        return _pairwise_distances_csr(X, Y, metric, n_jobs, **kwds)
    if issparse(X):
        return _pairwise_distances_csr(X, Y, metric, n_jobs, **kwds)
    if issparse(Y):
        return _pairwise_distances_csr(Y, X, metric, n_jobs, **kwds)
    if metric == 'precomputed':
        return _validate_precomputed(X, Y)
    else:
        return _pairwise_distances_numpy(X, Y, metric, n_jobs, **kwds)
    
    
    '''Razón para refactoring:

Quité las comprobaciones innecesarias elif y agregué comprobaciones isinstance para reducir la complejidad del código.
Cambié el orden de las condiciones para dar prioridad a los métodos basados en árboles y matrices dispersas.'''